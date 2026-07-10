import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from utils.utils import get_prompts


class LatentGRPO(nn.Module):
    """
    Implementation of LatentGRPO: Group Relative Policy Optimization 
    for Continuous Thought Reasoning.
    
    LatentGRPO freezes all parameters of the backbone LLM and introduces
    a single lightweight learnable projection module as the sole trainable component.
    """

    def __init__(self, config, llm_model_name, hidden_dim=None):
        """
        Initialize the LatentGRPO model.

        Args:
            config: Configuration name (e.g., 'small', 'mistral', 'qwen')
            llm_model_name: Name of the backbone LLM
            hidden_dim: Hidden dimension for the projection module (default: None, uses LLM's hidden size)
        """
        super().__init__()
        self.config = config
        self.llm_model_name = llm_model_name

        # Load the LLM model and tokenizer
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.llm_model.eval()
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        
        # Freeze all LLM parameters
        for p in self.llm_model.parameters():
            p.requires_grad = False

        # Get hidden dimension
        llm_hid_dim = self.llm_model.config.hidden_size if hidden_dim is None else hidden_dim

        # Initialize the projection module (2-layer MLP with LayerNorm)
        # Eq. (2) in the paper
        self.proj = nn.Sequential(
            nn.Linear(llm_hid_dim, llm_hid_dim),
            nn.GELU(),
            nn.Linear(llm_hid_dim, llm_hid_dim),
            nn.LayerNorm(llm_hid_dim)
        )

        # Reference projection parameters for KL divergence
        self.ref_proj = None

    def save_reference_projection(self):
        """Save a snapshot of the current projection parameters as reference."""
        self.ref_proj = {
            k: v.detach().clone() for k, v in self.proj.state_dict().items()
        }

    def get_input_embeddings(self, text, max_seq_len, device):
        """
        Get input embeddings from text.
        
        Args:
            text: Input text
            max_seq_len: Maximum sequence length
            device: Device to use
            
        Returns:
            Input embeddings
        """
        input_ids = self.llm_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )["input_ids"].to(device)
        embeddings = self.llm_model.get_input_embeddings()(input_ids)
        return embeddings, input_ids

    def generate_continuous_thoughts(self, query_embeddings, num_thoughts, device, noise_eps=None):
        """
        Generate continuous thought vectors recursively.
        
        Eq. (1) in the paper:
            h_k = LastHidden(LLM_phi([E_x; c_1; ...; c_{k-1}]))
            c_k = Proj_theta(h_k), k = 1, ..., K
        
        Args:
            query_embeddings: Input question embeddings (batch_size, seq_len, hidden_dim)
            num_thoughts: Number of continuous thought vectors K
            device: Device to use
            noise_eps: If provided, add noise to the first thought vector
            
        Returns:
            List of continuous thought vectors [c_1, c_2, ..., c_K]
        """
        thoughts = []
        current_embeddings = query_embeddings
        
        for k in range(num_thoughts):
            # Get last hidden state from LLM
            outputs = self.llm_model(
                inputs_embeds=current_embeddings,
                output_hidden_states=True,
            )
            # Get last hidden state
            h_k = outputs.hidden_states[-1][:, -1, :]  # (batch_size, hidden_dim)
            
            # Project to continuous thought vector
            c_k = self.proj(h_k)  # (batch_size, hidden_dim)
            
            # Add noise to first thought vector if provided
            if noise_eps is not None and k == 0:
                c_k = c_k + noise_eps
            
            thoughts.append(c_k)
            
            # Concatenate with current embeddings for next step
            # Expand c_k to match embedding dimensions
            c_k_expanded = c_k.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            current_embeddings = torch.cat([current_embeddings, c_k_expanded], dim=1)
        
        return thoughts

    def generate_answer(
        self,
        query_embeddings,
        thoughts,
        answer_text=None,
        max_gen_length=50,
        temperature=0.7,
    ):
        """
        Generate answer from query and continuous thoughts.
        
        Args:
            query_embeddings: Input question embeddings
            thoughts: List of continuous thought vectors
            answer_text: Ground truth answer (if provided, compute log probabilities)
            max_gen_length: Maximum generation length
            
        Returns:
            Generated answer and/or log probabilities
        """
        # Concatenate query embeddings, latent thoughts, and the answer prompt.
        thought_embeddings = torch.stack(thoughts, dim=1)  # (batch_size, K, hidden_dim)
        _, ans_prompt = get_prompts(self.config)
        ans_prompt_ids = self.llm_tokenizer(
            ans_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_gen_length,
            add_special_tokens=False,
        )["input_ids"].to(query_embeddings.device)
        ans_prompt_embeddings = self.llm_model.get_input_embeddings()(ans_prompt_ids)
        combined_embeddings = torch.cat(
            [query_embeddings, thought_embeddings, ans_prompt_embeddings], dim=1
        )
        
        if answer_text is not None:
            # Training mode: compute log probability of ground truth answer tokens.
            answer_ids = self.llm_tokenizer(
                answer_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_gen_length,
                add_special_tokens=False,
            )["input_ids"].to(query_embeddings.device)
            
            answer_embeddings = self.llm_model.get_input_embeddings()(answer_ids)
            
            # Concatenate answer embeddings
            full_embeddings = torch.cat([combined_embeddings, answer_embeddings], dim=1)
            
            attention_mask = torch.ones(
                (full_embeddings.shape[0], full_embeddings.shape[1]),
                dtype=torch.long,
                device=query_embeddings.device,
            )
            
            outputs = self.llm_model(
                inputs_embeds=full_embeddings,
                attention_mask=attention_mask,
            )
            logits = outputs.logits[:, :-1, :]

            labels = torch.full(
                (full_embeddings.shape[0], combined_embeddings.shape[1]),
                -100,
                dtype=torch.long,
                device=query_embeddings.device,
            )
            labels = torch.cat([labels, answer_ids], dim=1)
            target_ids = labels[:, 1:]
            
            # Compute answer-token negative log likelihood while ignoring the prefix.
            token_nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
                reduction='none'
            )
            token_nll = token_nll.view(target_ids.shape)
            answer_mask = target_ids.ne(-100)
            answer_nll = (token_nll * answer_mask).sum(dim=1)
            
            return -answer_nll
        else:
            # Inference mode: generate answer
            attention_mask = torch.ones(
                (combined_embeddings.shape[0], combined_embeddings.shape[1]),
                dtype=torch.long,
                device=query_embeddings.device,
            )
            
            with torch.no_grad():
                generated_ids = self.llm_model.generate(
                    inputs_embeds=combined_embeddings,
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen_length,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                )
            
            # Extract generated answer
            if generated_ids.shape[1] > combined_embeddings.shape[1]:
                answer_ids = generated_ids[:, combined_embeddings.shape[1]:]
            else:
                answer_ids = generated_ids
            answer = self.llm_tokenizer.decode(answer_ids[0], skip_special_tokens=True)
            
            return answer

    def sample_multi_trajectories(self, query_embeddings, num_thoughts, num_trajectories, device):
        """
        Sample multiple trajectories by injecting noise to the first thought vector.
        
        Eq. (3) in the paper:
            c_1^{(i)} = c_1 + epsilon^{(i)}, epsilon^{(i)} ~ N(0, I_d)
        
        Args:
            query_embeddings: Input question embeddings
            num_thoughts: Number of continuous thought vectors K
            num_trajectories: Number of trajectories G
            device: Device to use
            
        Returns:
            List of trajectories, each is a list of thought vectors
        """
        # Generate noise for each trajectory
        hidden_dim = query_embeddings.shape[-1]
        trajectories = []
        
        for i in range(num_trajectories):
            # Generate Gaussian noise
            noise = torch.randn(1, hidden_dim).to(device)
            
            # Generate trajectory with noise
            noisy_thoughts = self.generate_continuous_thoughts(
                query_embeddings, num_thoughts, device, noise_eps=noise
            )
            trajectories.append(noisy_thoughts)
        
        return trajectories

    def compute_contrastive_loss(self, trajectories, temperature=0.5):
        """
        Compute contrastive regularization loss to maintain trajectory diversity.
        
        Eq. (4) in the paper:
            L_cl = -sum_{i=1}^G log (exp(tau_i * tau_i / eta) / sum_{j=1}^G exp(tau_i * tau_j / eta))
        
        where tau_i = (1/K) * sum_{k=1}^K c_k^{(i)} is the aggregate representation.
        
        Args:
            trajectories: List of trajectories, each is a list of thought vectors
            temperature: Temperature parameter eta
            
        Returns:
            Contrastive loss
        """
        # Compute aggregate representation for each trajectory
        aggregates = []
        for traj in trajectories:
            # Stack thoughts and compute mean
            traj_stack = torch.stack(traj, dim=1)  # (1, K, hidden_dim)
            agg = traj_stack.mean(dim=1)  # (1, hidden_dim)
            aggregates.append(agg)
        
        # Stack all aggregates
        aggregates = torch.cat(aggregates, dim=0)  # (G, hidden_dim)
        
        # Compute similarity matrix
        similarity = torch.mm(aggregates, aggregates.t()) / temperature  # (G, G)
        
        # Compute InfoNCE loss
        # For each trajectory, its own representation should be most similar to itself
        labels = torch.arange(aggregates.shape[0]).to(aggregates.device)
        
        loss = F.cross_entropy(similarity, labels)
        
        return loss

    def compute_advantages(self, rewards):
        """
        Compute group-normalized relative advantages.
        
        Eq. (5) in the paper:
            A_i = (r_i - mean({r_j})) / std({r_j})
        
        Args:
            rewards: List of rewards for each trajectory
            
        Returns:
            Tensor of advantages
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std(unbiased=False).clamp_min(1e-8)
        
        advantages = (rewards_tensor - mean_reward) / std_reward
        
        return advantages

    def compute_policy_loss(self, log_probs, advantages, beta=0.1):
        """
        Compute policy optimization loss.
        
        Eq. (6) in the paper:
            L_LatentGRPO = -(1/G) * sum_{i=1}^G A_i * log p_phi(a_i | tau_i(theta), x) 
                          + beta * D_KL(pi_theta || pi_ref)
        
        Args:
            log_probs: Log probabilities of answers for each trajectory
            advantages: Advantages for each trajectory
            beta: KL divergence weight
            
        Returns:
            Policy loss
        """
        advantages = advantages.to(device=log_probs.device, dtype=log_probs.dtype)

        # Policy loss term
        policy_loss = -(advantages * log_probs).mean()
        
        # KL divergence term (simplified: use L2 distance between current and ref parameters)
        kl_loss = torch.zeros((), device=log_probs.device, dtype=log_probs.dtype)
        if self.ref_proj is not None:
            for name, param in self.proj.named_parameters():
                if name in self.ref_proj:
                    ref_param = self.ref_proj[name].to(
                        device=param.device, dtype=param.dtype
                    )
                    kl_loss = kl_loss + F.mse_loss(param, ref_param)
        
        total_loss = policy_loss + beta * kl_loss
        
        return total_loss, policy_loss, kl_loss

    def save_pretrained(self, path):
        """
        Save the LatentGRPO model to disk.

        Args:
            path: Path to save the model
        """
        os.makedirs(path, exist_ok=True)

        # Save the projection module
        torch.save(self.proj.state_dict(), os.path.join(path, "proj_module.pt"))

        # Save reference projection if exists
        if self.ref_proj is not None:
            torch.save(self.ref_proj, os.path.join(path, "ref_proj.pt"))

        # Save the config
        config = {
            "config": self.config,
            "llm_model_name": self.llm_model_name,
        }
        torch.save(config, os.path.join(path, "config.pt"))

    @classmethod
    def from_pretrained(cls, path):
        """
        Load a pretrained LatentGRPO model.

        Args:
            path: Path to the saved model

        Returns:
            Loaded LatentGRPO model
        """
        # Load config
        config = torch.load(os.path.join(path, "config.pt"))

        # Initialize model with loaded config
        model = cls(
            config["config"],
            config["llm_model_name"],
        )

        # Initialize and load the projection module
        model.proj.load_state_dict(
            torch.load(os.path.join(path, "proj_module.pt"), map_location="cpu")
        )
        
        # Load reference projection if exists
        ref_proj_path = os.path.join(path, "ref_proj.pt")
        if os.path.exists(ref_proj_path):
            model.ref_proj = torch.load(ref_proj_path, map_location="cpu")
        
        return model
