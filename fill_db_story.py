import json
from pprint import pprint

from chromadb import ClientAPI
from chromadb import Collection
import chromadb

from embedding_worker import TextEmbedder

embedder = TextEmbedder()
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

sentences = [
    'Посадил дед репку — выросла репка большая пребольшая',
    'Стал дед репку из земли тащить: тянет-потянет, вытащить не может.',
    'Позвал дед на помощь бабку',
    'Бабка за дедку, дедка за репку: тянут-потянут, вытянуть не могут.',
    'Позвала бабка внучку. Внучка за бабку, бабка за дедку, дедка за репку: тянут-потянут, вытянуть не могут.',
    'Кликнула внучка Жучку. Жучка за внучку, внучка за бабку, бабка за дедку, дедка за репку: тянут-потянут, вытянуть не могут.',
    'Кликнула Жучка кошку. Кошка за Жучку, Жучка за внучку, внучка за бабку, бабка за дедку, дедка за репку: тянут-потянут, вытянуть не могут.',
    'Кликнула кошка мышку.',
    'Мышка за кошку, кошка за Жучку, Жучка за внучку, внучка за бабку, бабка за дедку, дедка за репку тянут-потянут — вытащили репку!'
]

def create_collection(name : str, client: ClientAPI):
    return client.create_collection(name=name)


def fill_db_story(sentences: list[str], embedding_model: TextEmbedder, collection: Collection):
    for id, sentence in enumerate(sentences):
        collection.add(
            embeddings=embedding_model.text_to_embedding(sentence),
            ids=str(id),
            metadatas={'text': sentence}
        )

def fill_db_from_json(json_data: dict, embedding_model: TextEmbedder, collection: Collection):
    keys = list(json_data.keys())
    for ind, key in enumerate(keys):
        collection.add(embeddings=embedding_model.text_to_embedding(json_data[key]),
                       ids=key,
                       metadatas={
                           'text': json_data[key],
                           'article_number': key
                       })

    print(f'Данные добавлены в коллекцию: {collection.name}')


def get_full_info_sentence_embeddings(embedding: list[float], collection: Collection, n_results: int = 3):
    requests = collection.query(embedding, n_results=n_results)
    result_dict = {
        'distances': requests['distances'][0],
        'sentences': list([
            i['text']
            for i in requests['metadatas'][0]
        ])
    }

    # for ind, request in enumerate(requests):
    #     print(f'{len(request[ind])}')
    #     result_dict['distances'].append(request[0]['distances'][0][ind])
    #     result_dict['sentences'].append(request[0]['metadatas'][0][ind]['text'])

    return result_dict


def get_sentences_by_embedding(embedding: list[float], collection: Collection, n_results: int = 3) -> str:
    requests = collection.query(embedding, n_results=n_results)
    formatted_text = ''
    print("результат векторного поиска:")
    for ind, text in enumerate(requests['metadatas'][0]):
        print(ind,
              # text['article_number'],
              "\n", text['text'])
        t = text['text']
        formatted_text += f'{ind+1}. {t}\n'

    return formatted_text


if __name__ == '__main__':
    print(chroma_client.heartbeat())
    print(chroma_client.database)
    print(chroma_client.list_collections())
    pprint(list(chroma_client.get_settings()))
    collection = chroma_client.create_collection(
        name='test_collection',
        dimension=1024,
        metrics='cosine'
    )
    print(collection)

    print(chroma_client.list_collections())
    # collection = create_collection('fairy_tale', client=chroma_client)
    # fill_db_story(
    #     collection=collection,
    #     embedding_model=embedder,
    #     sentences=sentences)

    # collection = chroma_client.get_collection('fairy_tale')
    # pprint(
    #     collection.query(
    #         query_embeddings=embedder.text_to_embedding(
    #             'Кто позвал мышку?'
    #         ),
    #         n_results=3
    #     )
    # )
    #
    # result = get_sentences_by_embedding(
    #     embedding=embedder.text_to_embedding('Кто позвал мышку?'),
    #     collection=collection,
    #     n_results=4
    # )
    #
    # print(result)
    # pprint(result)
    # print('\n'.join(result))

    # cl = chroma_client.get_collection('ass')
    # cl = chroma_client.create_collection('ass')
    # cl.add(
    #     documents=['fuck the bull shit'],
    #     ids=['1'],
    #     metadatas={'text': 'fuck the bull shit'}
    # )
    #
    # data = cl.get(ids='1')
    # pprint(data)
    # print(data['metadatas'][0]['text'])
    # print(chroma_client.count_collections())
    # chroma_client.delete_collection('ass')
    # print(chroma_client.count_collections())

    # print(chroma_client.list_collections())
    # collection = chroma_client.create_collection('tk_rf')
    # collection = chroma_client.get_collection('tk_rf')

    # with open('tk_rf.json', 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    #     fill_db_from_json(data, collection=collection, embedding_model=embedder)
    #
    # print(chroma_client.list_collections())

    # pprint(collection.query(query_embeddings=embedder.text_to_embedding('как уволиться с работы?'), n_results=2))