import random
import time
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from models.latentgrpo import LatentGRPO
from utils.utils import clear_cache_in_dict, evaluate_pred, get_prompts


def process_batch(latentgrpo, batch_items, args, epoch=None):
    """
    Process a batch of items with multi-trajectory sampling and compute losses.
    
    Args:
        latentgrpo: LatentGRPO model
        batch_items: List of samples in the batch
        args: Arguments containing hyperparameters
        epoch: Current epoch number (for logging)
        
    Returns:
        Dictionary containing various losses
    """
    query_prompt, ans_prompt = get_prompts(args.config)
    
    # Hyperparameters
    K = args.train_max_contemp_tokens  # Number of continuous thought vectors
    G = getattr(args, 'num_trajectories', 4)  # Number of trajectories (G)
    lambda_cl = getattr(args, 'contrastive_lambda', 0.1)  # Contrastive loss weight
    beta = getattr(args, 'kl_beta', 0.1)  # KL divergence weight
    temperature = getattr(args, 'contrastive_temperature', 0.5)  # Temperature for contrastive loss
    
    total_policy_loss = 0
    total_contrastive_loss = 0
    total_kl_loss = 0
    
    for item in batch_items:
        # Prepare query
        query_text = query_prompt + item["query"]
        answer_text = item["answer"]
        
        # Get query embeddings
        query_embeddings, _ = latentgrpo.get_input_embeddings(
            query_text, args.max_seq_len, args.device
        )
        
        # Sample G trajectories
        trajectories = latentgrpo.sample_multi_trajectories(
            query_embeddings, K, G, args.device
        )
        
        # Compute contrastive loss (Eq. 4 in paper)
        contrastive_loss = latentgrpo.compute_contrastive_loss(trajectories, temperature)
        total_contrastive_loss += contrastive_loss.item()
        
        # Generate answers and compute log probabilities for each trajectory
        log_probs_list = []
        answers_list = []
        
        for traj in trajectories:
            # Compute log probability of ground truth answer
            log_prob = latentgrpo.generate_answer(
                query_embeddings, traj, answer_text=answer_text, max_gen_length=50
            )
            log_probs_list.append(log_prob)
            
            # Generate answer for reward computation
            with torch.no_grad():
                generated_answer = latentgrpo.generate_answer(
                    query_embeddings, traj, answer_text=None, max_gen_length=50
                )
            answers_list.append(generated_answer)
        
        # Compute rewards (Eq. 4 in paper: outcome-based reward)
        # r_i = K * 1[a_i = a*]
        rewards = []
        for ans in answers_list:
            correct = evaluate_pred(ans, answer_text, args.dataset)
            reward = K if correct else 0  # Total reward = K if correct, else 0
            rewards.append(reward)
        
        # Compute advantages (Eq. 5 in paper)
        advantages = latentgrpo.compute_advantages(rewards)
        
        # Compute policy loss (Eq. 6 in paper)
        log_probs_tensor = torch.cat(log_probs_list)
        total_loss, policy_loss, kl_loss = latentgrpo.compute_policy_loss(
            log_probs_tensor, advantages, beta=beta
        )
        
        total_policy_loss += policy_loss.item()
        total_kl_loss += kl_loss.item()
        
        # Total loss (Eq. 7 in paper)
        # L = L_LatentGRPO + lambda * L_cl
        batch_loss = total_loss + lambda_cl * contrastive_loss
        
        # Backward pass
        batch_loss.backward()
        
        # Clear cache
        del query_embeddings, trajectories, log_probs_list, answers_list
        torch.cuda.empty_cache()
    
    # Return average losses
    num_items = len(batch_items)
    return {
        'policy_loss': total_policy_loss / num_items,
        'contrastive_loss': total_contrastive_loss / num_items,
        'kl_loss': total_kl_loss / num_items,
    }


def train_latentgrpo_model(logger, args, train_dataset, eval_dataset, lr=1e-4, wd=0.01):
    """
    Train the LatentGRPO model's projection module.
    """
    num_epochs = getattr(args, 'latentgrpo_epochs', 10)
    
    latentgrpo = LatentGRPO(args.config, args.teacher_model_name).to(args.device)
    
    # Save initial projection parameters as reference for KL divergence
    latentgrpo.save_reference_projection()
    
    # Only train the projection module
    optimizer = optim.AdamW(latentgrpo.proj.parameters(), lr=lr, weight_decay=wd)
    
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        latentgrpo.proj.train()
        
        # Training loop
        epoch_losses = {
            'policy_loss': 0,
            'contrastive_loss': 0,
            'kl_loss': 0,
        }
        
        # Shuffle training data
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        
        batch_size = int(args.batch_size)
        for batch_start in tqdm(
            range(0, len(indices), batch_size),
            desc=f"Epoch {epoch + 1}/{num_epochs} - Training",
        ):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_items = [train_dataset[idx] for idx in batch_indices]
            
            optimizer.zero_grad()
            
            # Process batch
            losses = process_batch(latentgrpo, batch_items, args, epoch)
            
            # Update weights
            optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            
            torch.cuda.empty_cache()
        
        # Compute average training losses
        num_batches = (len(indices) + batch_size - 1) // batch_size
        avg_train_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        # Validation
        latentgrpo.proj.eval()
        val_accuracy = run_validation(latentgrpo, eval_dataset, args)
        
        # Log metrics
        logger.log_metrics(
            {
                **{f"train_{k}": v for k, v in avg_train_losses.items()},
                "val_accuracy": val_accuracy,
            },
            epoch
        )
        
        logger.logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Policy Loss: {avg_train_losses['policy_loss']:.6f}, "
            f"Contrastive Loss: {avg_train_losses['contrastive_loss']:.6f}, "
            f"KL Loss: {avg_train_losses['kl_loss']:.6f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            latentgrpo.save_pretrained(args.model_save_path)
            logger.logger.info(
                f"Saved best model with validation accuracy: {best_val_accuracy:.4f}"
            )
    
    del latentgrpo
    torch.cuda.empty_cache()
    return args.model_save_path


def run_validation(latentgrpo, eval_dataset, args):
    """
    Run validation on the evaluation dataset.
    
    Args:
        latentgrpo: LatentGRPO model
        eval_dataset: Evaluation dataset
        args: Arguments
        
    Returns:
        Validation accuracy
    """
    query_prompt, ans_prompt = get_prompts(args.config)
    K = getattr(args, 'eval_max_contemp_tokens', 1)  # Use fewer thoughts for evaluation
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sample in tqdm(eval_dataset, desc="Running validation"):
            # Prepare query
            query_text = query_prompt + sample["query"]
            
            # Get query embeddings
            query_embeddings, _ = latentgrpo.get_input_embeddings(
                query_text, args.max_seq_len, args.device
            )
            
            # Generate thoughts (deterministic, no noise)
            thoughts = latentgrpo.generate_continuous_thoughts(
                query_embeddings, K, args.device, noise_eps=None
            )
            
            # Generate answer
            generated_answer = latentgrpo.generate_answer(
                query_embeddings, thoughts, answer_text=None, max_gen_length=50
            )
            
            # Evaluate
            if evaluate_pred(generated_answer, sample["answer"], args.dataset):
                correct += 1
            total += 1
            
            # Clear cache
            del query_embeddings, thoughts
            torch.cuda.empty_cache()
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def run_latentgrpo_inference(logger, latentgrpo, dataset, args):
    """
    Run inference with the LatentGRPO model.
    
    Args:
        logger: Logger object
        latentgrpo: LatentGRPO model
        dataset: Dataset to evaluate on
        args: Arguments
        
    Returns:
        List of metrics for different temperatures
    """
    all_metrics = []
    query_prompt, ans_prompt = get_prompts(args.config)
    K = getattr(args, 'eval_max_contemp_tokens', 1)
    
    for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results = []
        
        latentgrpo.proj.eval()
        with torch.no_grad():
            for sample in tqdm(dataset, desc=f"Running inference (temp={temp})"):
                contemp_start = time.time()
                
                # Prepare query
                query_text = query_prompt + sample["query"]
                
                # Get query embeddings
                query_embeddings, _ = latentgrpo.get_input_embeddings(
                    query_text, args.max_seq_len, args.device
                )
                contemp_time = time.time() - contemp_start
                
                # Generate thoughts (deterministic, no noise)
                gen_start = time.time()
                thoughts = latentgrpo.generate_continuous_thoughts(
                    query_embeddings, K, args.device, noise_eps=None
                )
                
                # Generate answer with specified temperature
                # Note: Need to temporarily modify generation parameters
                generated_answer = latentgrpo.generate_answer(
                    query_embeddings,
                    thoughts,
                    answer_text=None,
                    max_gen_length=50,
                    temperature=temp,
                )
                gen_time = time.time() - gen_start
                
                # Evaluate
                is_correct = evaluate_pred(generated_answer, sample["answer"], dataset.name)
                
                results.append({
                    "query": sample["query"],
                    "correct": int(is_correct),
                    "sample_time": contemp_time + gen_time,
                })
                
                # Clear cache
                del query_embeddings, thoughts
                torch.cuda.empty_cache()
        
        # Compute metrics
        metrics = {
            "numerical_accuracy": float(np.mean([r["correct"] for r in results])),
            "ave_sample_time": float(np.mean([r["sample_time"] for r in results])),
            "dataset": args.dataset,
            "eval_temp": temp,
            "train_max_contemp_tokens": args.train_max_contemp_tokens,
            "eval_max_contemp_tokens": K,
        }
        
        logger.logger.info(
            f"eval_temp = {temp} | acc = {metrics['numerical_accuracy']:.4f} | "
            f"ave_sample_time = {metrics['ave_sample_time']:.4f}"
        )
        
        all_metrics.append(metrics)
    
    return all_metrics
