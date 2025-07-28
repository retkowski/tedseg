import nltk
import pandas as pd
import requests
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

nltk.download('punkt_tab')

def get_transcript(video_id):
    url = "https://www.ted.com/graphql"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-operation-name": "Transcript",
    }
    data = {
        "query": """
          query GetTranslation($language: String!, $videoId: ID!) {
            translation(language: $language, videoId: $videoId) {
              paragraphs {
                cues {
                  text
                  time
                }
              }
            }
          }
        """,
        "variables": {"videoId": video_id, "language": "en"},
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching {video_id}: {response.status_code}, {response.text}")
        return None

def deep_len(l):
    lengths = [len(ll) for ll in l]
    return sum(lengths), lengths

def get_target(sections):
    length, section_lengths = deep_len(sections)
    target_sequence = "".join(['1' + '0' * (x - 1) for x in section_lengths])
    target_sequence = '0' + target_sequence[1:]
    return target_sequence

def process_partition(partition):
    print(f"\nProcessing partition: {partition}")
    input_file = f"talk_ids.{partition}.json"
    output_file = f"transcripts.{partition}.json"

    df = pd.read_json(input_file)
    talk_ids = df.squeeze().tolist() if isinstance(df, pd.DataFrame) else df.tolist()

    tokenized_data = {}

    for talk_id in tqdm(talk_ids, desc=f"Downloading transcripts for {partition}"):
        transcript_data = get_transcript(talk_id)
        if not transcript_data or not transcript_data.get('data') or not transcript_data['data'].get('translation'):
            print(f"Skipping {talk_id} due to missing transcript.")
            continue

        paragraphs = transcript_data['data']['translation']['paragraphs']
        paragraph_texts = [' '.join([cue['text'] for cue in p['cues']]).replace('\n', ' ') for p in paragraphs]

        sentence_blocks = [sent_tokenize(p) for p in paragraph_texts]

        target = get_target(sentence_blocks)

        flattened_sentences = [sent for para in sentence_blocks for sent in para]

        tokenized_data[talk_id] = {
            'talk_id': talk_id,
            'text': flattened_sentences,
            'targets': '|=' + target
        }

    df_out = pd.DataFrame.from_dict(tokenized_data, orient='index').reset_index(drop=True)
    df_out['targets'] = df_out['targets'].astype("string")
    df_out.to_json(output_file, orient='records', indent=2)
    print(f"Saved processed data to {output_file}")

    return df_out

all_dfs = {}
for part in ['test']:
    df = process_partition(part)
    all_dfs[part] = df

print("\nAll partitions processed.")
