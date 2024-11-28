import itertools
import os
import pickle
import traceback 
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn.functional as F

from captum.attr import configure_interpretable_embedding_layer
from utils import load_csv_dataset
from utils import load_pickle_dataset, load_from_hf, load_from_mila

import itertools
import torch.nn.functional as F
import pickle
from pathlib import Path
from tqdm import tqdm
import traceback 

def get_dataset(dataset, num_passes: int = 1):
    for i in range(num_passes):
        for sample in dataset:
            yield sample

@torch.no_grad()
def compute_attributions(
    model,
    tokenizer,
    proteins,
    device: torch.device,
    save_file: Path,
    chunk_size: int = 128,
    mask_probability: float = 0.15,
    span_probability: float = None,
    span_max: int = 1,
    masked_only: bool = True,
    exclude_special_tokens_replacement: bool = False,
    fp16: bool = True,
    max_length: int = None,
    objects: list = None,
):  
    # TF32 and FP16
    
    replacement_ids = torch.ones((len(tokenizer)))
    if exclude_special_tokens_replacement:
        for i in tokenizer.special_token_ids:
            replacement_ids[i] = 0

    # file = open(save_file, 'wb')
    # try:
    
    with open(save_file, 'wb') as file:

        if not (objects is None):
            for pass_data in objects:
                pickle.dump(pass_data, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=fp16):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            for i, (label, protein) in enumerate(proteins):
               
                # Tokenize the protein and decode to get the special tokens
                x = torch.as_tensor(tokenizer.encode(protein)).to(torch.long)
                protein = tokenizer.decode(x, skip_special_tokens=False).split()

                if len(protein) > max_length:
                    continue
                
                masked_ids = torch.full(size=(len(protein),), fill_value=False, dtype=torch.bool)
                
                # Compute the padding, <bos> and <eos> masks
                pad_mask = x == tokenizer.pad_token_id
                bos_mask = x == tokenizer.bos_token_id
                eos_mask = x == tokenizer.eos_token_id
            
                # MLM
                if span_probability is None or span_max is None or span_probability == 0 or span_max == 1:
                    probability_matrix = torch.full(x.shape, mask_probability)
                    probability_matrix.masked_fill_(pad_mask | bos_mask | eos_mask, value=0.0)
                    masked_ids = torch.bernoulli(probability_matrix).bool()
            
                # Span masking
                else:
                    uniform_dist = torch.distributions.uniform.Uniform(0, len(protein))
                    geometric_dist = torch.distributions.geometric.Geometric(span_probability)
                    while torch.sum(masked_ids) / len(protein) < mask_probability:
                        span_start = int(uniform_dist.sample().item())
                        span_length = int(min(geometric_dist.sample().item(), span_max - 1, len(protein) - span_start))
                        masked_ids[span_start : span_start + span_length + 1] = True
                        # Unmask the padding, <bos> and <eos> tokens (note that padding should not be necessary)
                        masked_ids = masked_ids & ~pad_mask & ~bos_mask & ~eos_mask
            
                y = x.clone()
    
                # Replace only with mask
                if masked_only:
                    x[masked_ids] = tokenizer.mask_token_id
    
                # Replace with both mask and random words
                else:
                    # 80% of the time, the masked input tokens are replaced with <MASK>
                    replaced_ids = torch.bernoulli(torch.full(y.shape, 0.8)).bool() & masked_ids
                    x[replaced_ids] = tokenizer.mask_token_id
                
                    # 10% of the time, the masked input tokens are replaced with a random word
                    random_ids = torch.bernoulli(torch.full(y.shape, 0.5)).bool() & masked_ids & ~replaced_ids
                    random_words = torch.multinomial(replacement_ids, torch.numel(x), replacement=True).view(x.size())
                    x[random_ids] = random_words[random_ids]
    
                # Get reference output (relevant positions are replaced)
                reference_input = x.unsqueeze(0)
                reference_input = reference_input.to(device)
                reference_embeddings = interpretable_embedding.indices_to_embeddings(reference_input)
                reference_output = model(reference_embeddings).logits
    
                # Get masked outputs
                mask_indices = masked_ids.nonzero()
    
                # Create test sequences (each sequence has a single mask value replaced with its real value)
                test_input = reference_input.repeat(mask_indices.size(0), 1)
                for index, value in enumerate(mask_indices):
                    test_input[index, value] = y[value]
    
                # Assert that our replacement matches
                assert test_input[0, mask_indices[0][0]] == y[mask_indices[0][0]], "Masking isn't correct!"
    
                # Create embeddings for the testing seqeunces and get the logits from a forward pass
                test_embeddings = interpretable_embedding.indices_to_embeddings(test_input)
                test_output = model(test_embeddings).logits
    
                # Compute attributions in terms of the KL divergence and the absolute change in logits
                mask_indices = mask_indices.squeeze()

                # If there's only one index that is masked, just skip it
                if mask_indices.shape[0] <= 2:
                    print(f"Skipped protein {label} with length {len(protein)}.")

                # Get all pairs of masked indices (all permuations)
                pairs = itertools.permutations(torch.arange(mask_indices.shape[0]).detach().tolist(), 2)
        
                tot_distance = len(protein)
    
                pass_data = {
                    'label': label,
                    'model': model_name,
                    'mask_indices': mask_indices.tolist(),
                    'masked_index': [],
                    'target_index': [],
                    'absolute_difference': [], 
                    'kl_div': [], 
                    'distance': [], 
                    'tot_distance': tot_distance
                }
            
                for (modified_index, target_index) in pairs:
        
                    distance = abs(mask_indices[target_index] - mask_indices[modified_index]).item()
                    
                    original_logits = reference_output[0][mask_indices[target_index]]
                    new_logits = test_output[modified_index][mask_indices[target_index]]
            
                    absolute_difference = torch.mean(torch.abs(original_logits - new_logits)).detach().item()
            
                    log_softmax_reference = F.log_softmax(original_logits, dim=0)
                    softmax_targets = F.softmax(new_logits, dim=0)
            
                    kl_div = F.kl_div(log_softmax_reference, softmax_targets, reduction="none").mean().detach().item()
    
                    pass_data['masked_index'].append(modified_index)
                    pass_data['target_index'].append(target_index)
                    pass_data['absolute_difference'].append(absolute_difference)
                    pass_data['kl_div'].append(kl_div)
                    pass_data['distance'].append(distance)
                    
                pickle.dump(pass_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    # except:
    #     file.close()
    #     traceback.print_exc() 
    #     return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source", 
        type=string,
        help="Type of model.",
        default="mila",
        choices=["mila", "hf"]
    )

    parser.add_argument(
        "--model_name", 
        type=string,
        help="Name of model.",
        default="AMPLIFY350M",
    )

    parser.add_argument(
        "--model_path", 
        type=string,
        help="Path to model weights.",
        default="/network/projects/drugdiscovery/AMPLIFY/checkpoints/AMPLIFY_350M/pytorch_model.pt",
    )

    parser.add_argument(
        "--tokenizer_path", 
        type=string,
        help="Path to tokenizer. Only specify if different from the path to the model.",
        default=None,
    )

    parser.add_argument(
        "--config_path", 
        type=string,
        help="Path to config file.",
        default=Path(os.environ.get('AMPLIFY_PROJECT_PATH')) / "data"/ "AMPLIFY_350_config.yaml",
    )

    parser.add_argument(
        "--device", 
        type=string,
        help="Type of device to run inference on.",
        default="cuda",
        choices=["cuda", "cpu"]
    )

    parser.add_argument(
        "--compile", 
        action="store_true",
        help="Whether or not to compile functions.",
    )

    parser.add_argument(
        "--fp16", 
        action="store_true",
        help="Whether or not to use half-precision.",
    )

    parser.add_argument(
        "--n_proteins", 
        type=int,
        default=None,
        help="Maximum number of proteins to compute attributions on.",
    )

    parser.add_argument(
        "--dataset", 
        type=string,
        default="UniProt",
        choices=["UniProt", "CASP14"],
        help="Dataset to compute attributions on.",
    )

    parser.add_argument(
        "--num_passes", 
        type=int,
        default=5,
        help="Number of passes or times to compute attributions on the same protein.",
    )

    parser.add_argument(
        "--chunk_size", 
        type=int,
        default=128,
        help="Maximum batch size to use when computing attributions.",
    )

    parser.add_argument(
        "--mask_probability", 
        type=float,
        default=0.15,
        help="Masking probability for attribution computation.",
    )

    parser.add_argument(
        "--span_probability", 
        type=float,
        default=None,
        help="Probability for span masking.",
    )

    parser.add_argument(
        "--span_max", 
        type=int,
        default=1,
        help="Maximum length for a span if using span masking. Only useful if `span_probability` is set to a value between 0 and 1.",
    )

    parser.add_argument(
        "--exclude_special_tokens_replacement", 
        action="store_true",
        help="Do not replace special tokens with masks when doing masking for attribution computation."
    )

    parser.add_argument(
        "--masked_only", 
        action="store_true",
        help="Replace only with mask token for attribution computation.",
    )

    parser.add_argument(
        "--save_folder", 
        type=string,
        default="/home",
        help="Path to the folder to save attribution results. Must provide a value.",
        required=True,
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        required=True, 
        help="The seed to use for computing attributions. Defaults to 42."
    )
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)

    save_file = Path(args.save_folder) / "masked_influence_attribution" / f"{args.model_name}-{args.dataset}-{args.num_passes}-{args.seed}.pickle"

    # Model
    source = args.source # "mila"
    model_name = args.model_name # "AMPLIFY350M"
    model_path = args.model_path # "/network/projects/drugdiscovery/AMPLIFY/checkpoints/AMPLIFY_350M/pytorch_model.pt"
    tokenizer_path = args.tokenizer_path # None 
    config_path = args.config_path # "AMPLIFY_350_config.yaml"
    device = args.device # "cuda"
    compile = args.compile # False
    fp16 = args.fp16 # True

    # Get model and tokenizer
    if source == "hf":
        model, tokenizer = load_from_hf(model_path, tokenizer_path, fp16=fp16)
        interpretable_embedding = configure_interpretable_embedding_layer(model, "esm.embeddings.word_embeddings")
        bos_id, mask_id, eos_id = tokenizer.cls_token_id, tokenizer.mask_token_id, tokenizer.eos_token_id
        max_length = model.config.max_position_embeddings
    elif source == "mila":
        model, tokenizer = load_from_mila(model_path, config_path)
        interpretable_embedding = configure_interpretable_embedding_layer(model, "encoder")
        bos_id, mask_id, eos_id = tokenizer.bos_token_id, tokenizer.mask_token_id, tokenizer.eos_token_id
        max_length = model.config.max_length
    else:
        raise Exception("Only 'hf' and 'mila' sources are supported, not {source}.")
        
    model.to(device)
    
    # Dataset
    n_proteins = args.n_proteins # 1_000_000
    
    dataset = args.dataset
    
    if dataset == "CASP14":
        data_name = "CASP14"
        data_path = os.environ.get('AMPLIFY_PROJECT_PATH') / "data"/ "casp14.pickle"
    
        # Load dataset
        labels, proteins, dist_matrices = load_pickle_dataset(data_path, n_proteins, max_length)
        full_dataset = list(zip(labels, proteins))
        
        for (label, protein) in itertools.islice(full_dataset, 0, 5):
            print(label, protein)
        
    else:
        # Dataset
        data_name = "UniProt"
        data_path = os.environ.get('AMPLIFY_PROJECT_PATH') / "data"/ "uniprot_dev.csv"
        
        # Prepare the dataset
        full_dataset = load_csv_dataset(data_path, n_proteins)
        
        for (label, protein) in itertools.islice(full_dataset, 0, 5):
            print(label, protein)
    
    num_passes = args.num_passes # 5
    
    count = 0
    prior_objects = []
    if os.path.exists(save_file):
        file = open(save_file, 'rb')
        
        while True:
            try:
                # This needs to be fixed, but I need to find a way such that I can stream to a file and append to it without overwriting existing data. This is a very dirty workaround but will need to look at alternative methods (it's just that pickle files are easier to work with than other formats).
                prior_objects.append(pickle.load(file))
                count += 1
            except EOFError:
                break
    
    print(f"Skipping {count} elements.")
    
    repeated_dataset = get_dataset(full_dataset, num_passes)
    sliced_dataset = itertools.islice(repeated_dataset, count, None)

    compute_attributions(
        model=model,
        tokenizer=tokenizer,
        proteins=sliced_dataset,
        device=device,
        save_file=save_file,
        chunk_size=args.chunk_size,
        mask_probability=args.mask_probability,
        span_probability=args.span_probability,
        span_max=args.span_max,
        masked_only=args.masked_only,
        exclude_special_tokens_replacement=args.exclude_special_tokens_replacement,
        fp16=fp16,
        max_length=max_length,
        objects=prior_objects
    )