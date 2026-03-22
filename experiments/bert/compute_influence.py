import argparse

import torch.nn.functional as F
from accelerate import Accelerator
from utils import construct_model, get_loaders

import logix
import torch



def main():
    parser = argparse.ArgumentParser("GLUE Influence Analysis")
    parser.add_argument("--project", type=str, default="sst2")
    parser.add_argument("--config_path", type=str, default="./config.yaml")
    parser.add_argument("--data_name", type=str, default="sst2")
    parser.add_argument("--damping", type=float, default=None)
    args = parser.parse_args()

    # prepare model & data loader
    model, tokenizer = construct_model(
        args.data_name, ckpt_path=f"files/checkpoints/0/{args.data_name}_epoch_3.pt"
    )
    model.eval()
    test_loader = get_loaders(data_name=args.data_name)[-1]

    accelerator = Accelerator()
    model, test_loader = accelerator.prepare(model, test_loader)

    run = logix.init(args.project, config=args.config_path)


    if run is None:
         print("ERROR: logix.init() returned None. Was it already initialized globally?")
         return
    elif hasattr(run, 'influence'):
        print("DEBUG: run.influence attribute FOUND after init.")
    else:
        print("DEBUG: run.influence attribute MISSING after init. This is unexpected.")

        print("DEBUG: Attempting to manually add influence attribute...")
        from logix.analysis import InfluenceFunction
        try:
            run.influence = InfluenceFunction(state=run.state)
            if hasattr(run, 'influence'):
                 print("DEBUG: Manual addition successful.")
            else:
                 print("ERROR: Manual addition failed.")
        except Exception as e:
             print(f"ERROR during manual addition: {e}")



    logix.watch(model)
    logix.initialize_from_log()
    log_loader = logix.build_log_dataloader()


    logix.setup({"grad": ["log"]})
    logix.eval()
    for batch in test_loader:
        data_id = tokenizer.batch_decode(batch["input_ids"])
        labels = batch.pop("labels").view(-1)
        _ = batch.pop("idx")
        with run(data_id=data_id, mask=batch["attention_mask"]):
            model.zero_grad()
            outputs = model(**batch)
            logits = outputs.view(-1, outputs.shape[-1])
            loss = F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)
            accelerator.backward(loss)

        test_log = run.get_log()

        result = run.influence.compute_influence_all(test_log, log_loader, damping=args.damping)


        if result and "influence" in result:
            if_scores = result["influence"]
            if not isinstance(if_scores, torch.Tensor):
                if_scores = torch.tensor(if_scores) # Convert list to tensor perhaps


            save_path = "if_logix.pt"
            torch.save(if_scores, save_path)
            print(f"Plaintext influence scores saved to {save_path}")


            if if_scores.ndim > 1: # Handle potential multiple test samples
                scores_to_rank = if_scores[0] # Rank for the first test sample
            else:
                scores_to_rank = if_scores
            _, top_influential_data = torch.topk(scores_to_rank, k=10)
            print("Top 10 Plaintext influential data indices:", top_influential_data.cpu().numpy().tolist())

        else:
            print("ERROR: Influence computation did not return expected results.")

        break


if __name__ == "__main__":
    main()
