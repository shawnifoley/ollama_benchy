import argparse
import time
from datetime import datetime
from typing import List, Optional

import ollama


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class OllamaResponse:
    def __init__(
        self,
        model: str,
        created_at: datetime,
        done: bool,
        total_duration: float,
        load_duration: int = 0,
        prompt_eval_count: int = -1,
        prompt_eval_duration: int = 0,
        eval_count: int = 0,
        eval_duration: int = 0,
        message: Optional[Message] = None,
    ):
        self.model = model
        self.created_at = created_at
        self.done = done
        self.total_duration = total_duration
        self.load_duration = load_duration
        self.prompt_eval_count = prompt_eval_count
        self.prompt_eval_duration = prompt_eval_duration
        self.eval_count = eval_count
        self.eval_duration = eval_duration
        self.message = message


def run_benchmark(
    model_name: str, prompt: str, verbose: bool
) -> Optional[OllamaResponse]:
    start_time = time.time()
    full_response = ""
    last_element = None
    try:
        response_gen = ollama.generate(model=model_name, prompt=prompt, stream=True)
        for response in response_gen:
            if verbose:
                print(response["response"], end="", flush=True)
            full_response += response.get("response", "")
            last_element = response

        end_time = time.time()
        total_duration = end_time - start_time
        if last_element:
            return OllamaResponse(
                model=last_element.get("model"),
                created_at=last_element.get("created_at"),
                done=last_element.get("done"),
                total_duration=total_duration if total_duration is not None else 0.0,
                load_duration=last_element.get("load_duration", 0),
                prompt_eval_count=last_element.get("prompt_eval_count", -1),
                prompt_eval_duration=last_element.get("prompt_eval_duration", 0),
                eval_count=last_element.get("eval_count", 0),
                eval_duration=last_element.get("eval_duration", 0),
                message=Message(role="assistant", content=full_response),
            )
        else:
            return None
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return None


def to_sec(nanosecs):
    return nanosecs / 1000000000


def inference_stats(model_response: OllamaResponse):
    prompt_ts = model_response.prompt_eval_count / (
        to_sec(model_response.prompt_eval_duration)
    )
    response_ts = model_response.eval_count / (to_sec(model_response.eval_duration))
    total_ts = (model_response.prompt_eval_count + model_response.eval_count) / (
        to_sec(model_response.prompt_eval_duration + model_response.eval_duration)
    )

    print(
        f"""
----------------------------------------------------
        {model_response.model}
        \tPrompt eval: {prompt_ts:.2f} t/s
        \tResponse: {response_ts:.2f} t/s
        \tTotal: {total_ts:.2f} t/s

        Stats:
        \tPrompt tokens: {model_response.prompt_eval_count}
        \tResponse tokens: {model_response.eval_count}
        \tModel load time: {to_sec(model_response.load_duration):.2f}s
        \tPrompt eval time: {to_sec(model_response.prompt_eval_duration):.2f}s
        \tResponse time: {to_sec(model_response.eval_duration):.2f}s
        \tTotal time: {model_response.total_duration:.2f}s
----------------------------------------------------
        """
    )


def average_stats(responses: List[OllamaResponse]):
    if not responses:
        print("No stats to average")
        return

    res = OllamaResponse(
        model=responses[0].model,
        created_at=datetime.now(),
        message=Message(
            role="system",
            content=f"Average stats across {len(responses)} runs",
        ),
        done=True,
        total_duration=sum(r.total_duration for r in responses),
        load_duration=sum(r.load_duration for r in responses),
        prompt_eval_count=sum(r.prompt_eval_count for r in responses),
        prompt_eval_duration=sum(r.prompt_eval_duration for r in responses),
        eval_count=sum(r.eval_count for r in responses),
        eval_duration=sum(r.eval_duration for r in responses),
    )
    print("Average stats:")
    inference_stats(res)


def get_benchmark_models() -> List[str]:
    models = ollama.list().get("models", [])
    if isinstance(models, dict) and "models" in models:
        models = models["models"]
    model_names = [model.model for model in models]
    return model_names


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks on your Ollama models."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output",
        default=False,
    )
    parser.add_argument(
        "-p",
        "--prompts",
        nargs="*",
        default=[
            "Why is the sky blue?",
            "Write a story about a robot that paints houses.",
        ],
        help="Custom prompots to use for benchmarking. Separate multiple prompts with spaces. Default: Why is the sky blue? Write a story about a robot that paints houses.",
    )

    args = parser.parse_args()

    verbose = args.verbose
    prompts = args.prompts
    print(f"\nPrompts: {prompts}")

    model_names = get_benchmark_models()
    if not model_names:
        print("No Ollama models found.")
        return

    print("Available models:")
    for i, model_name in enumerate(model_names):
        print(f"{i + 1}. {model_name}")

    while True:
        try:
            selection = int(input("Select a model to benchmark (enter number): "))
            if 1 <= selection <= len(model_names):
                selected_model = model_names[selection - 1]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print(f"Benchmarking model: {selected_model}\n")

    benchmarks = {}
    responses: List[OllamaResponse] = []
    for prompt in prompts:
        if verbose:
            print(f"\nPrompt: {prompt}")
        response = run_benchmark(selected_model, prompt, verbose=verbose)
        if response:
            responses.append(response)

        if verbose and response:
            print(f"Response: {response.message.content}")
            inference_stats(response)

    benchmarks[selected_model] = responses

    for model_name, responses in benchmarks.items():
        average_stats(responses)


if __name__ == "__main__":
    main()
    # Example usage:
    # python benchmark.py --verbose --prompts "What color is the sky" "Write a story about a robot that paints houses."
