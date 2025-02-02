from utils.load_data import load_beir_data
def main():

    dataset_list =["msmarco", "trec-covid", "nfcorpus", "nq", "hotpotqa", "fiqa", "arguana", "webis-touche2020", "quora" ,"dbpedia-entity" ,"scidocs", "fever", "climate-fever", "scifact"] ##"cqadupstack",
    for sub_dataset in dataset_list:
        corpus, queries, qrels = load_beir_data(sub_dataset, 'test')
        print('load :', sub_dataset)

if __name__ == '__main__':
    main()