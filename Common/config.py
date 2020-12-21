import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice")

    parser.add_argument("--world", default=1, type=int,
                        help="The id number of the mario world.")
    parser.add_argument("--stage", default=1, type=int,
                        help="The id number of the mario world's stage.")
    parser.add_argument("--total_iterations", default=6000, type=int,
                        help="The total number of iterations.")
    parser.add_argument("--interval", default=10, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by iterations.")
    parser.add_argument("--do_train", action="store_false",
                        help="The flag determines whether to train the agent or play with it.")
    parser.add_argument("--render", action="store_true",
                        help="The flag determines whether to render each agent or not.")
    parser.add_argument("--train_from_scratch", action="store_false",
                        help="The flag determines whether to train from scratch or continue previous tries.")

    parser_params = parser.parse_args()

    # region default parameters
    default_params = {"state_shape": (4, 84, 84),
                      "rollout_length": 128,
                      "n_epochs": 4,
                      "batch_size": 64,
                      "lr": 2.5e-4,
                      "gamma": 0.99,
                      "lambda": 0.95,
                      "ent_coeff": 0.01,
                      "clip_range": 0.2,
                      "n_workers": 8,
                      "max_grad_norm": 0.5
                      }

    # endregion
    total_params = {**vars(parser_params), **default_params}
    print("params:", total_params)
    return total_params
