"""This is the main module to launch the training of the different experiments
with the T5 model."""

from absl import app, flags

from src.reference_implementations.prompt_zoo.data_utility import create_semeval_sentiment_dataset
from src.reference_implementations.prompt_zoo.model_utility import run_model
from src.reference_implementations.prompt_zoo.prompted_t5 import PromptedT5

FLAGS = flags.FLAGS
flags.DEFINE_string("task_name", "semeval_3_class_sentiment", "the name of the downstream nlp task.")
flags.DEFINE_string("train_file", "/tmp/train.csv", "the path/name of the train file.")


def launch_no_prompt_train() -> None:
    """launch the training phase for the no prompting experiments for the
    following cases with the T5 model.

    1 - Fully fine-tuning all the parameters of the T5 model.
    2 - Only fine-tuning the shared input embedding layer of the T5 encoder/decoder.
    3 - Only fine-tuning the output embedding layer of the T5 decoder.
    4 - Fine-tuning both the shared input embedding layer +
        the output embedding layer of the T5 decoder.
    """

    FLAGS.mode = "train"

    prompted_t5_model = PromptedT5()
    if FLAGS.task_name == "semeval_3_class_sentiment":
        dataloader = create_semeval_sentiment_dataset(
            tokenizer=prompted_t5_model.tokenizer, file_name=FLAGS.train_file, shuffle=True
        )
        run_model(model=prompted_t5_model, train_dataloader=dataloader)
    return


def main(argv) -> None:  # type: ignore
    """Main function to switch over the t5 experiment type and launch the
    correct train script."""
    if FLAGS.t5_exp_type in ["all_finetune", "input_finetune" "output_finetune", "input_output_finetune"]:
        launch_no_prompt_train()
    return


if __name__ == "__main__":
    app.run(main)
