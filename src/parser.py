def generate_argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--train",action="store_true",help="generate PPO and DQN model save")
    parser.add_argument("-p","--policy",required=True,choices=["PPO","DQN"],help="choose the policy to play with")
    parser.add_argument("-c","--case-study",required=True,choices=["1","2","3"],help="Choose the case study")
    return parser