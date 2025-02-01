import argparse
import os

# some times I use config.py because it is easy to use and understand

def parse():
    parser = argparse.ArgumentParser(description='Unsupervised Corpus Poisoning')

    #for train reconstruction.py
    parser.add_argument('--dataset', type=str, default="msmarco",
                        help='BEIR dataset to train, nq, nq-train, msmarco  arguana fiqa')
    parser.add_argument('--split', type=str, default='test',
                        help='train, test , dev')
    parser.add_argument('--recon_model_code', type=str, default='contriever-msmarco', # we train it's ctx encoder
                        help='contriever, contriever-msmarco, dpr-single, dpr-multi, ance, tas-b, dragon, retromae_msmarco, retromae_msmarco_finetune, retromae_msmarco_distill ' )
    parser.add_argument('--decoder_base_model', type=str, default="google-bert/bert-base-uncased",help='google-bert/bert-base-uncased , google-bert/bert-large-uncased')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--max_query_length', type=int, default=32)
    parser.add_argument('--pad_to_max_length', default=True)
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--output_file", default=None, type=str)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=50000)


    # for retrieval evaluation on Beir dataset and attack
    parser.add_argument("--eval_dataset", type=str, default="nfcorpus", help='BEIR dataset to evaluate, nq , scifact, fiqa, arguana,msmarco  nfcorpus')
    parser.add_argument('--eval_model_code', type=str, default='simlm-msmarco', help='contriever, contriever-msmarco, simlm-msmarco, dpr-single, dpr-multi, ance, tas-b, dragon, retromae_msmarco, retromae_msmarco_finetune, retromae_msmarco_distill ')


    # for attack
    parser.add_argument("--method", type=str, default='unsupervised', help="the hotflip method to attack the corpus")
    parser.add_argument('--attack_dataset', type=str,default='nfcorpus',
                        help='attack_dataset to  attack. If not provided, it will default to the value of --dataset nq-train, nq, msmarco, arguana, fiqa, nfcorpus')
    parser.add_argument('--attack_number', type=int, default=100, help='the number of attack')
    parser.add_argument('--attack_mode_code', type=str, default='simlm-msmarco', help='contriever, contriever-msmarco, dpr-single, dpr-multi, ance, tas-b, dragon, retromae_msmarco, retromae_msmarco_finetune, retromae_msmarco_distill ')

    parser.add_argument("--attack_model_path", type=str, default="output/models/facebook_contriever/Tevatron_msmarco-passage-aug/robust_lambda-0.0_seed-2024_bs-32_qlen-32_plen-128_groupsize-16/checkpoint-35000", help='attack model\'s path' )
    parser.add_argument("--k", default=10, type=int,help="k means to attack")
    parser.add_argument("--num_cand", default=100, type=int)  # # Top-100 tokens as candidates
    parser.add_argument("--num_iter", default=5000, type=int)
    parser.add_argument("--num_adv_passage_tokens", default=50, type=int)
    parser.add_argument("--init_gold", dest='init_gold', action='store_true', help="If specified, init with gold passages")
    parser.add_argument("--no_init_gold", dest='init_gold', action='store_false', help="If specified, do not init with gold passages")
    parser.add_argument("--do_kmeans", default=True, action="store_true")
    parser.add_argument('--attack_query', default=True, help="whether attack the query or documents")
    parser.add_argument("--num_grad_iter", default=1, type=int)

    args = parser.parse_args()
    return args