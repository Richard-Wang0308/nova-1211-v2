# PARALLEL MINER: 3 GPU Workers with Shared Generator & Pool
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import json
import time
import bittensor as bt
import pandas as pd
from pathlib import Path
import nova_ph2
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from nova_ph2.PSICHIC.psichic_utils.data_utils import virtual_screening

from molecules import (
    generate_valid_random_molecules_batch,
    select_diverse_elites,
    build_component_weights,
)

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")

# ============================================================================
# GPU WORKER PROCESS
# ============================================================================
def gpu_worker(worker_id: int, config: dict, input_queue: Queue, output_queue: Queue):
    """
    GPU worker process that loads models once and scores batches of molecules.
    
    Args:
        worker_id: Unique ID for this worker (0, 1, 2)
        config: Configuration dict
        input_queue: Queue to receive molecule batches
        output_queue: Queue to send scored results
    """
    bt.logging.info(f"[Worker {worker_id}] Starting GPU worker...")
    
    # Initialize models for this worker
    target_models = []
    antitarget_models = []
    
    try:
        # Load target models
        for seq in config["target_sequences"]:
            wrapper = PsichicWrapper()
            wrapper.initialize_model(seq)
            target_models.append(wrapper)
        
        # Load antitarget models
        for seq in config["antitarget_sequences"]:
            wrapper = PsichicWrapper()
            wrapper.initialize_model(seq)
            antitarget_models.append(wrapper)
        
        bt.logging.info(f"[Worker {worker_id}] Models loaded: {len(target_models)} targets, {len(antitarget_models)} antitargets")
    
    except Exception as e:
        bt.logging.error(f"[Worker {worker_id}] Failed to load models: {e}")
        output_queue.put(("ERROR", worker_id, str(e)))
        return
    
    # Worker loop: receive batches, score them, send results back
    while True:
        try:
            # Get batch from queue (blocking)
            message = input_queue.get()
            
            # Check for termination signal
            if message == "STOP":
                bt.logging.info(f"[Worker {worker_id}] Received STOP signal, shutting down")
                break
            
            iteration, batch_df = message
            batch_start = time.time()
            
            if batch_df.empty:
                bt.logging.warning(f"[Worker {worker_id}] Received empty batch in iteration {iteration}")
                output_queue.put((iteration, worker_id, pd.DataFrame()))
                continue
            
            smiles_list = batch_df["smiles"].tolist()
            
            # Score against target models
            target_scores = []
            for target_model in target_models:
                scores = target_model.score_molecules(smiles_list)
                
                # Share smiles_dict with antitarget models
                for antitarget_model in antitarget_models:
                    antitarget_model.smiles_list = smiles_list
                    antitarget_model.smiles_dict = target_model.smiles_dict
                
                scores.rename(columns={'predicted_binding_affinity': "target"}, inplace=True)
                target_scores.append(scores["target"])
            
            # Average target scores
            target_series = pd.DataFrame(target_scores).mean(axis=0)
            
            # Score against antitarget models
            antitarget_scores = []
            for i, antitarget_model in enumerate(antitarget_models):
                antitarget_model.create_screen_loader(
                    antitarget_model.protein_dict, 
                    antitarget_model.smiles_dict
                )
                antitarget_model.screen_df = virtual_screening(
                    antitarget_model.screen_df,
                    antitarget_model.model,
                    antitarget_model.screen_loader,
                    os.getcwd(),
                    save_interpret=False,
                    ligand_dict=antitarget_model.smiles_dict,
                    device=antitarget_model.device,
                    save_cluster=False,
                )
                scores = antitarget_model.screen_df['predicted_binding_affinity']
                antitarget_scores.append(scores)
            
            # Average antitarget scores
            anti_series = pd.DataFrame(antitarget_scores).mean(axis=0)
            
            # Add scores to batch
            result_df = batch_df.copy()
            result_df["Target"] = target_series.values
            result_df["Anti"] = anti_series.values
            result_df["score"] = result_df["Target"] - (config["antitarget_weight"] * result_df["Anti"])
            
            batch_time = time.time() - batch_start
            bt.logging.info(
                f"[Worker {worker_id}] Iter {iteration}: Scored {len(result_df)} molecules in {batch_time:.2f}s"
            )
            
            # Send results back
            output_queue.put((iteration, worker_id, result_df))
        
        except Exception as e:
            bt.logging.error(f"[Worker {worker_id}] Error processing batch: {e}")
            output_queue.put(("ERROR", worker_id, str(e)))


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================
def get_config(input_file: str = os.path.join(BASE_DIR, "input.json")):
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}


def main(config: dict, num_workers: int = 3):
    """
    Main orchestrator that manages shared pool and distributes work to GPU workers.
    
    Args:
        config: Configuration dict
        num_workers: Number of parallel GPU workers (default: 3)
    """
    bt.logging.info(f"Starting parallel miner with {num_workers} GPU workers")
    
    # Configuration
    n_samples_per_worker = config["num_molecules"] * 5  # 500 molecules per worker
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    
    # Shared state
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score", "Target", "Anti"])
    top_pool = top_pool.astype({'score': 'float64', 'Target': 'float64', 'Anti': 'float64'})
    seen_inchikeys = set()
    
    # Dynamic parameters
    mutation_prob = 0.1
    elite_frac = 0.25
    
    # Statistics
    iteration = 0
    start = time.time()
    total_time = 0
    total_requested = 0
    total_unique = 0
    score_improvement_rate = 0.0
    
    # Create queues for communication
    input_queues = [Queue() for _ in range(num_workers)]
    output_queue = Queue()
    
    # Start worker processes
    workers = []
    for worker_id in range(num_workers):
        p = Process(
            target=gpu_worker,
            args=(worker_id, config, input_queues[worker_id], output_queue)
        )
        p.start()
        workers.append(p)
        bt.logging.info(f"Started worker {worker_id} (PID: {p.pid})")
    
    # Wait for workers to initialize
    time.sleep(5)
    bt.logging.info("All workers initialized, starting main loop")
    
    # First iteration uses more samples
    n_samples_first_iteration = n_samples_per_worker * num_workers * 4 if config["allowed_reaction"] != "rxn:5" else n_samples_per_worker * num_workers
    
    try:
        # Main iteration loop
        while time.time() - start < 1800:  # 30 minutes
            iteration += 1
            iter_start_time = time.time()
            
            # Build component weights from top pool
            component_weights = None
            if not top_pool.empty:
                component_weights = build_component_weights(top_pool, rxn_id)
            
            # Select elites for crossover/mutation
            elite_df = pd.DataFrame()
            elite_names = None
            if not top_pool.empty:
                elite_df = select_diverse_elites(top_pool, min(100, len(top_pool)))
                elite_names = elite_df["name"].tolist() if not elite_df.empty else None
            
            # Generate molecules for ALL workers
            total_samples = n_samples_first_iteration if iteration == 1 else (n_samples_per_worker * num_workers)
            
            data = generate_valid_random_molecules_batch(
                rxn_id,
                n_samples=total_samples,
                db_path=DB_PATH,
                subnet_config=config,
                batch_size=300,
                elite_names=elite_names,
                elite_frac=elite_frac,
                mutation_prob=mutation_prob,
                avoid_inchikeys=seen_inchikeys,
                component_weights=component_weights,
            )
            
            gen_time = time.time() - iter_start_time
            bt.logging.info(
                f"[Main] Iteration {iteration}: Generated {len(data)} molecules in {gen_time:.2f}s"
            )
            
            if data.empty:
                bt.logging.warning(f"[Main] Iteration {iteration}: No valid molecules generated")
                continue
            
            total_requested += len(data)
            
            # Filter out seen molecules
            filtered_data = data[~data["InChIKey"].isin(seen_inchikeys)]
            total_unique += len(filtered_data)
            
            if len(filtered_data) < len(data):
                bt.logging.info(
                    f"[Main] Iteration {iteration}: Filtered {len(data) - len(filtered_data)} seen molecules"
                )
            
            # Adjust parameters based on duplication rate
            dup_ratio = (len(data) - len(filtered_data)) / max(1, len(data))
            if dup_ratio > 0.6:
                mutation_prob = min(0.5, mutation_prob * 1.5)
                elite_frac = max(0.2, elite_frac * 0.8)
            elif dup_ratio < 0.2 and not top_pool.empty:
                mutation_prob = max(0.05, mutation_prob * 0.9)
                elite_frac = min(0.8, elite_frac * 1.1)
            
            data = filtered_data.reset_index(drop=True)
            
            if data.empty:
                bt.logging.warning(f"[Main] Iteration {iteration}: All molecules were duplicates")
                continue
            
            # Split data into batches for workers
            batch_size = len(data) // num_workers
            batches = []
            for i in range(num_workers):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size if i < num_workers - 1 else len(data)
                batch = data.iloc[start_idx:end_idx].copy()
                batches.append(batch)
            
            bt.logging.info(
                f"[Main] Iteration {iteration}: Split into {num_workers} batches: " +
                ", ".join([f"{len(b)} mols" for b in batches])
            )
            
            # Send batches to workers
            gpu_start_time = time.time()
            for worker_id, batch in enumerate(batches):
                input_queues[worker_id].put((iteration, batch))
            
            # Collect results from all workers
            results = []
            errors = []
            for _ in range(num_workers):
                result = output_queue.get()  # Blocking wait
                if result[0] == "ERROR":
                    errors.append(result)
                else:
                    results.append(result)
            
            if errors:
                bt.logging.error(f"[Main] Iteration {iteration}: {len(errors)} workers reported errors")
                for error in errors:
                    bt.logging.error(f"  Worker {error[1]}: {error[2]}")
            
            if not results:
                bt.logging.error(f"[Main] Iteration {iteration}: No valid results from workers")
                continue
            
            # Combine results from all workers
            all_scored = pd.concat([r[2] for r in results], ignore_index=True)
            
            gpu_time = time.time() - gpu_start_time
            bt.logging.info(
                f"[Main] Iteration {iteration}: GPU scoring completed in {gpu_time:.2f}s "
                f"({len(all_scored)} molecules from {len(results)} workers)"
            )
            
            # Update seen molecules
            seen_inchikeys.update(all_scored["InChIKey"].tolist())
            
            # Store previous average before updating pool
            prev_iter_avg_score = top_pool['score'].mean() if len(top_pool) > 0 else None
            
            # Update top pool
            total_data = all_scored[["name", "smiles", "InChIKey", "score", "Target", "Anti"]].copy()
            total_data = total_data.astype({'score': 'float64', 'Target': 'float64', 'Anti': 'float64'})
            
            top_pool = pd.concat([top_pool, total_data], ignore_index=True)
            top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
            top_pool = top_pool.sort_values(by="score", ascending=False)
            top_pool = top_pool.head(config["num_molecules"])
            
            current_avg_score = top_pool['score'].mean() if len(top_pool) > 0 else None
            
            # Calculate improvement
            if current_avg_score is not None and prev_iter_avg_score is not None:
                score_improvement_rate = (current_avg_score - prev_iter_avg_score) / max(abs(prev_iter_avg_score), 1e-6)
            else:
                score_improvement_rate = 0.0
            
            iter_total_time = time.time() - iter_start_time
            total_time += iter_total_time
            
            improvement_pct = 0.0
            if prev_iter_avg_score is not None and current_avg_score is not None and prev_iter_avg_score != 0:
                improvement_pct = (current_avg_score - prev_iter_avg_score) / abs(prev_iter_avg_score)
            
            # Statistics
            avg_score = float(top_pool['score'].mean()) if len(top_pool) > 0 else 0.0
            max_score = float(top_pool['score'].max()) if len(top_pool) > 0 else 0.0
            min_score = float(top_pool['score'].min()) if len(top_pool) > 0 else 0.0
            
            # Log progress
            print(
                f"[Main] Iter {iteration} | "
                f"Workers: {num_workers} | "
                f"Avg: {avg_score:.6f} | Best: {max_score:.6f} | Min: {min_score:.6f} | "
                f"Improvement: {improvement_pct*100:+.2f}% | "
                f"Time: {iter_total_time:.2f}s | Total: {total_time:.2f}s | "
                f"Pool: {len(top_pool)} | Scored: {len(all_scored)} | "
                f"Requested: {total_requested} | Unique: {total_unique} | "
                f"Elite: {elite_frac:.2f} | Mut: {mutation_prob:.2f}"
            )
            
            # Save results
            top_entries = {"molecules": top_pool["name"].tolist()}
            with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
                json.dump(top_entries, f, ensure_ascii=False, indent=2)
    
    finally:
        # Cleanup: stop all workers
        bt.logging.info("Stopping all workers...")
        for worker_id in range(num_workers):
            input_queues[worker_id].put("STOP")
        
        # Wait for workers to finish
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                bt.logging.warning(f"Worker {worker.pid} did not stop gracefully, terminating")
                worker.terminate()
        
        bt.logging.info("All workers stopped")
        
        # Final statistics
        bt.logging.info("="*80)
        bt.logging.info("FINAL STATISTICS")
        bt.logging.info(f"Total iterations: {iteration}")
        bt.logging.info(f"Total molecules evaluated: {total_unique}")
        bt.logging.info(f"Total molecules requested: {total_requested}")
        bt.logging.info(f"Unique ratio: {total_unique/max(1, total_requested):.2%}")
        bt.logging.info(f"Best score: {top_pool['score'].max():.6f}")
        bt.logging.info(f"Pool size: {len(top_pool)}")
        bt.logging.info("="*80)


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    config = get_config()
    bt.logging.info("Configuration loaded")
    
    # Run with 3 workers (can be adjusted based on GPU memory)
    main(config, num_workers=3)