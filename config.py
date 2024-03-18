import argparse
def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--user_query", type=str,
                        default="", help="openai-keys")
        args = parser.parse_args()
        return args