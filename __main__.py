import argparse
import json
import subprocess
from datetime import datetime
import uuid
import pandas as pd
import os
from . import common

from .browsecomp_eval import BrowseCompEval
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .healthbench_eval import HealthBenchEval
from .healthbench_meta_eval import HealthBenchMetaEval
from .humaneval_eval import HumanEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .simpleqa_eval import SimpleQAEval
from .sampler.ensemble_grader_sampler import EnsembleGraderSampler
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS
from .sampler.o_chat_completion_sampler import OChatCompletionSampler
from .sampler.responses_sampler import ResponsesSampler
from .sampler.ollama_sampler import OllamaSampler


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--list-evals", action="store_true", help="List available evaluations"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Select a model by name. Also accepts a comma-separated list of models.",
    )
    parser.add_argument(
        "--grader-model",
        type=str,
        default=None,
        help="Comma-separated list of models to use for grading.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        help="Select an eval by name. Also accepts a comma-separated list of evals.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of repeats to run. Only supported for certain evals.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=120,
        help="Number of threads to run. Only supported for HealthBench and HealthBenchMeta.",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["none", "minimal", "low", "medium", "high"],
        default=None,
        help="Reasoning effort for reasoning models (none/minimal=fastest, low=faster, medium=balanced, high=best quality). GPT-5.1 uses 'none', GPT-5 uses 'minimal'. Applies to GPT-5, GPT-5.1, o3, o4-mini, etc.",
    )

    args = parser.parse_args()

    # Use lambda functions for lazy initialization to avoid API key checks when listing models
    available_models = {
        # Ollama Models
        "qwen34b": lambda: OllamaSampler(model="qwen3:4b", max_tokens=2048),
        "qwen38b": lambda: OllamaSampler(model="qwen3:8b", max_tokens=2048),
        "llama3.2": lambda: OllamaSampler(model="llama3.2:1b", max_tokens=2048),
        "llama3.1": lambda: OllamaSampler(model="llama3.1:8b", max_tokens=2048),
        "gemma3": lambda: OllamaSampler(model="gemma3:latest", max_tokens=2048),
        "gemma327b": lambda: OllamaSampler(model="gemma3:27b", max_tokens=2048),
        "medgemma4b": lambda: OllamaSampler(model="alibayram/medgemma:4b", max_tokens=2048),
        "medgemma27b": lambda: OllamaSampler(model="alibayram/medgemma:27b", max_tokens=2048),
        # Reasoning Models
        "o3": lambda: ResponsesSampler(
            model="o3-2025-04-16",
            reasoning_model=True,
        ),
        "o3-temp-1": lambda: ResponsesSampler(
            model="o3-2025-04-16",
            reasoning_model=True,
            temperature=1.0,
        ),
        "o3_high": lambda: ResponsesSampler(
            model="o3-2025-04-16",
            reasoning_model=True,
            reasoning_effort="high",
        ),
        "o3_low": lambda: ResponsesSampler(
            model="o3-2025-04-16",
            reasoning_model=True,
            reasoning_effort="low",
        ),
        # Default == Medium
        "o4-mini": lambda: ResponsesSampler(
            model="o4-mini-2025-04-16",
            reasoning_model=True,
        ),
        "o4-mini_high": lambda: ResponsesSampler(
            model="o4-mini-2025-04-16",
            reasoning_model=True,
            reasoning_effort="high",
        ),
        "o4-mini_low": lambda: ResponsesSampler(
            model="o4-mini-2025-04-16",
            reasoning_model=True,
            reasoning_effort="low",
        ),
        "o1-pro": lambda: ResponsesSampler(
            model="o1-pro",
            reasoning_model=True,
        ),
        "o1": lambda: OChatCompletionSampler(
            model="o1",
        ),
        "o1_high": lambda: OChatCompletionSampler(
            model="o1",
            reasoning_effort="high",
        ),
        "o1_low": lambda: OChatCompletionSampler(
            model="o1",
            reasoning_effort="low",
        ),
        "o1-preview": lambda: OChatCompletionSampler(
            model="o1-preview",
        ),
        "o1-mini": lambda: OChatCompletionSampler(
            model="o1-mini",
        ),
        # Default == Medium
        "o3-mini": lambda: OChatCompletionSampler(
            model="o3-mini",
        ),
        "o3-mini_high": lambda: OChatCompletionSampler(
            model="o3-mini",
            reasoning_effort="high",
        ),
        "o3-mini_low": lambda: OChatCompletionSampler(
            model="o3-mini",
            reasoning_effort="low",
        ),
        # GPT-5 models - All GPT-5 models are reasoning models using ResponsesSampler
        "gpt-5": lambda: ResponsesSampler(
            model="gpt-5",
            reasoning_model=True,
            reasoning_effort=args.reasoning_effort,
        ),
        "gpt-5-2025-02-27": lambda: ResponsesSampler(
            model="gpt-5-2025-02-27",
            reasoning_model=True,
            reasoning_effort=args.reasoning_effort,
        ),
        "gpt-5-pro": lambda: ResponsesSampler(
            model="gpt-5-pro",
            reasoning_model=True,
            reasoning_effort=args.reasoning_effort,
        ),
        "gpt-5-mini": lambda: ResponsesSampler(
            model="gpt-5-mini",
            reasoning_model=True,
            reasoning_effort=args.reasoning_effort or "low",  # Default to low for mini
        ),
        "gpt-5-nano": lambda: ResponsesSampler(
            model="gpt-5-nano",
            reasoning_model=True,
            reasoning_effort=args.reasoning_effort or "low",  # Default to low for nano
        ),
        # GPT-5.1 models - All GPT-5.1 models are reasoning models
        "gpt-5.1": lambda: ResponsesSampler(
            model="gpt-5.1",
            reasoning_model=True,
            reasoning_effort=args.reasoning_effort,
        ),
        "gpt-5.1-2025-04-14": lambda: ResponsesSampler(
            model="gpt-5.1-2025-04-14",
            reasoning_model=True,
            reasoning_effort=args.reasoning_effort,
        ),
        # GPT-4.1 models
        "gpt-4.1": lambda: ChatCompletionSampler(
            model="gpt-4.1-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4.1-temp-1": lambda: ChatCompletionSampler(
            model="gpt-4.1-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            temperature=1.0,
        ),
        "gpt-4.1-mini": lambda: ChatCompletionSampler(
            model="gpt-4.1-mini-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4.1-nano": lambda: ChatCompletionSampler(
            model="gpt-4.1-nano-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # GPT-4o models
        "gpt-4o": lambda: ChatCompletionSampler(
            model="gpt-4o",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-2024-11-20": lambda: ChatCompletionSampler(
            model="gpt-4o-2024-11-20",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-2024-08-06": lambda: ChatCompletionSampler(
            model="gpt-4o-2024-08-06",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-2024-08-06-temp-1": lambda: ChatCompletionSampler(
            model="gpt-4o-2024-08-06",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            temperature=1.0,
        ),
        "gpt-4o-2024-05-13": lambda: ChatCompletionSampler(
            model="gpt-4o-2024-05-13",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-mini": lambda: ChatCompletionSampler(
            model="gpt-4o-mini-2024-07-18",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # GPT-4.5 model
        "gpt-4.5-preview": lambda: ChatCompletionSampler(
            model="gpt-4.5-preview-2025-02-27",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # GPT-4-turbo model
        "gpt-4-turbo-2024-04-09": lambda: ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        ),
        # GPT-4 model
        "gpt-4-0613": lambda: ChatCompletionSampler(
            model="gpt-4-0613",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        ),
        # GPT-3.5 Turbo model
        "gpt-3.5-turbo-0125": lambda: ChatCompletionSampler(
            model="gpt-3.5-turbo-0125",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        ),
        "gpt-3.5-turbo-0125-temp-1": lambda: ChatCompletionSampler(
            model="gpt-3.5-turbo-0125",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1.0,
        ),
        # Chatgpt models:
        "chatgpt-4o-latest": lambda: ChatCompletionSampler(
            model="chatgpt-4o-latest",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            max_tokens=2048,
        ),
        "gpt-4-turbo-2024-04-09_chatgpt": lambda: ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        ),
        # Claude models - Latest Claude 4.5 series
        "claude-sonnet-4-5": lambda: ClaudeCompletionSampler(
            model="claude-sonnet-4-5-20250929",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
        "claude-haiku-4-5": lambda: ClaudeCompletionSampler(
            model="claude-haiku-4-5-20251001",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
        "claude-opus-4-1": lambda: ClaudeCompletionSampler(
            model="claude-opus-4-1-20250805",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
        # Claude legacy models
        "claude-3-opus": lambda: ClaudeCompletionSampler(
            model="claude-3-opus-20240229",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
        "claude-3-5-sonnet": lambda: ClaudeCompletionSampler(
            model="claude-3-5-sonnet-20241022",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
        "claude-3-sonnet": lambda: ClaudeCompletionSampler(
            model="claude-3-sonnet-20240229",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
        "claude-3-haiku": lambda: ClaudeCompletionSampler(
            model="claude-3-haiku-20240307",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
    }

    if args.list_models:
        print("Available models:")
        for model_name in available_models.keys():
            print(f" - {model_name}")
        return
    
    if args.list_evals:
        print("Available evaluations:")
        print(" - mmlu")
        print(" - math (needs equality_checker)")
        print(" - gpqa")
        print(" - mgsm")
        print(" - drop")
        print(" - humaneval")
        print(" - simpleqa (requires --grader-model)")
        print(" - browsecomp (requires --grader-model)")
        print(" - healthbench (requires --grader-model)")
        print(" - healthbench_hard (requires --grader-model)")
        print(" - healthbench_consensus (requires --grader-model)")
        print(" - healthbench_meta")
        return

    if args.model:
        models_chosen = args.model.split(",")
        for model_name in models_chosen:
            if model_name not in available_models:
                print(f"Error: Model '{model_name}' not found.")
                return

        if args.eval == "healthbench_meta":
            if len(models_chosen) == 1:
                models = {model_name: available_models[models_chosen[0]]()}
            else:
                models_list = [
                    available_models[model_name]() for model_name in models_chosen
                ]
                ensemble_sampler = EnsembleGraderSampler(models_list)
                ensemble_name = "-".join(models_chosen)
                models = {ensemble_name: ensemble_sampler}
        else:
            models = {
                model_name: available_models[model_name]() for model_name in models_chosen
            }

    print(f"Running with args {args}")

    grading_sampler = None
    if args.grader_model:
        graders_chosen = args.grader_model.split(",")
        invalid = [g for g in graders_chosen if g not in available_models]
        if invalid:
            print(f"Error: Grader model(s) {invalid} not found.")
            return

        grader_samplers = [available_models[g]() for g in graders_chosen]
        if len(grader_samplers) == 1:
            grading_sampler = grader_samplers[0]
            grader_label = graders_chosen[0]
        else:
            grading_sampler = EnsembleGraderSampler(grader_samplers)
            grader_label = "ensemble_" + "-".join(graders_chosen)
    
    # Create equality checker for math eval
    equality_checker = lambda: ChatCompletionSampler(
        model="gpt-4o-2024-11-20",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
    )

    def get_evals(eval_name, debug_mode, grading_sampler):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "mmlu":
                return MMLUEval(num_examples=1 if debug_mode else num_examples)
            case "math":
                return MathEval(
                    equality_checker=equality_checker(),
                    num_examples=num_examples,
                    n_repeats=1 if (debug_mode or num_examples is not None) else args.n_repeats or 10,
                )
            case "gpqa":
                return GPQAEval(
                    n_repeats=1 if debug_mode else args.n_repeats or 10,
                    num_examples=num_examples,
                )
            case "mgsm":
                return MGSMEval(
                    num_examples_per_lang=10 if debug_mode else num_examples or 250
                )
            case "drop":
                return DropEval(
                    num_examples=10 if debug_mode else num_examples,
                    train_samples_per_prompt=3,
                )
            case "humaneval":
                return HumanEval(num_examples=10 if debug_mode else num_examples)
            case "simpleqa":
                return SimpleQAEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case "browsecomp":
                return BrowseCompEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case "healthbench":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name=None,
                )
            case "healthbench_hard":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name="hard",
                )
            case "healthbench_consensus":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name="consensus",
                )
            case "healthbench_meta":
                return HealthBenchMetaEval(
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    if args.eval:
        evals_list = args.eval.split(",")
        evals = {}
        for eval_name in evals_list:
            try:
                evals[eval_name] = get_evals(eval_name, args.debug, grading_sampler)
            except Exception as e:
                error_msg = str(e)
                # Check if it's a known eval that's missing requirements
                if eval_name in ["healthbench", "healthbench_hard", "healthbench_consensus", "simpleqa", "browsecomp"]:
                    if grading_sampler is None:
                        print(f"Error: eval '{eval_name}' requires --grader-model to be specified.")
                        print(f"Example: --eval={eval_name} --model=gpt-4o --grader-model=gpt-4o")
                    elif "404" in error_msg or "ResourceNotFound" in error_msg or "No such file" in error_msg:
                        print(f"Error: The data file for '{eval_name}' is not available.")
                        print(f"Try downloading it first using: just download-{eval_name}")
                    else:
                        print(f"Error initializing eval '{eval_name}': {error_msg}")
                elif eval_name == "healthbench_meta":
                    print(f"Error initializing eval '{eval_name}': {error_msg}")
                else:
                    print(f"Error: eval '{eval_name}' not found.")
                    print(f"Available evals: mmlu, math, gpqa, mgsm, drop, humaneval, simpleqa, browsecomp, healthbench, healthbench_hard, healthbench_consensus, healthbench_meta")
                return
    else:
        evals = {
            eval_name: get_evals(eval_name, args.debug, grading_sampler)
            for eval_name in [
                "healthbench",
                "healthbench_hard",
                "healthbench_consensus",
                "healthbench_meta",
            ]
        }

    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    print(f"Running the following evals: {list(evals.keys())}")
    print(f"Running evals for the following models: {list(models.keys())}")
    if args.grader_model:
        print(f"Using grader: {grader_label}")
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(base_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    run_dir = os.path.join(tmp_dir, f"{date_str}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            if args.grader_model:
                file_stem = f"{eval_name}_{model_name}_grader-{grader_label}"
            else:
                file_stem = f"{eval_name}_{model_name}"
            # file stem should also include the year, month, day, and time in hours and minutes
            file_stem += f"_{date_str}"
            report_filename = os.path.join(run_dir, f"{file_stem}{debug_suffix}.html")
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w", encoding="utf-8") as fh:
                fh.write(common.make_report(result))
            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            # Sort metrics by key
            metrics = dict(sorted(metrics.items()))
            print(metrics)
            result_filename = os.path.join(run_dir, f"{file_stem}{debug_suffix}.json")
            with open(result_filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")

            full_result_filename = os.path.join(
                run_dir, f"{file_stem}{debug_suffix}_allresults.json"
            )
            with open(full_result_filename, "w", encoding="utf-8") as f:
                result_dict = {
                    "score": result.score,
                    "metrics": result.metrics,
                    "htmls": result.htmls,
                    "convos": result.convos,
                    "metadata": result.metadata,
                }
                f.write(json.dumps(result_dict, indent=2))
                print(f"Writing all results to {full_result_filename}")

            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
