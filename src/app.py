# app.py
# Streamlit POS Tag Visualization using HMM
# Author: Bilal Malik

import streamlit as st
import joblib
import numpy as np
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import random
from nltk.corpus import brown
from nltk.tag.hmm import HiddenMarkovModelTrainer

# ------------------------------------------------------------
# SETUP
# ------------------------------------------------------------
nltk.download('brown')
nltk.download('universal_tagset')

# Load and preprocess data
tagged_sentences = brown.tagged_sents(tagset='universal')
tagged_sentences = [[(word.lower(), tag) for (word, tag) in sent]
                    for sent in tagged_sentences]

# Load saved model
model = joblib.load("hmm_model_partial.pkl")

# Split data for supervised HMM tagging (for visualization)
random.seed(42)
random.shuffle(tagged_sentences)
split_idx = int(0.9 * len(tagged_sentences))
train_sents = tagged_sentences[:split_idx]

trainer = HiddenMarkovModelTrainer()
hmm_tagger = trainer.train_supervised(train_sents)

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def tag_sentence(sentence):
    """Return tagged list of (word, tag)."""
    return hmm_tagger.tag(sentence)


def top_n_tags(word, n=3, viterbi_tag=None):
    """Return top N tags for visualization (includes Viterbi tag)."""
    states = list(hmm_tagger._states)
    tags = []
    if viterbi_tag:
        tags.append(viterbi_tag)
    for tag in states:
        if tag != viterbi_tag and len(tags) < n:
            tags.append(tag)
    random.shuffle(tags)  # shuffle so correct tag isn't always on top
    return tags


def build_columnar_viterbi_graph(sentence, top_n=3):
    """Builds a NetworkX graph showing top tags per word with clean layout."""
    G = nx.DiGraph()
    tagged = hmm_tagger.tag(sentence)
    viterbi_tags = [tag for _, tag in tagged]

    node_positions = {}
    nodes_per_column = []

    for i, word in enumerate(sentence):
        v_tag = viterbi_tags[i]
        tags = top_n_tags(word, top_n, viterbi_tag=v_tag)
        nodes = []

        for j, tag in enumerate(tags):
            node_id = f"{i}_{tag}"
            G.add_node(node_id, label=f"{tag}\n'{word}'")

            # Better layout:
            # - even vertical spacing (no overlap)
            # - slight horizontal jitter for natural appearance
            x = i + random.uniform(-0.15, 0.15)
            y = (top_n - j) * 1.3
            node_positions[node_id] = (x, y)
            nodes.append(node_id)
        nodes_per_column.append(nodes)

    # Connect nodes between consecutive columns
    for col in range(len(nodes_per_column) - 1):
        for src in nodes_per_column[col]:
            for tgt in nodes_per_column[col + 1]:
                src_tag = src.split("_")[1]
                tgt_tag = tgt.split("_")[1]
                if src_tag == viterbi_tags[col] and tgt_tag == viterbi_tags[col + 1]:
                    G.add_edge(src, tgt, color='green', weight=2)
                else:
                    G.add_edge(src, tgt, color='gray', weight=0.6)

    return G, node_positions, tagged


def plot_columnar_viterbi(G, node_positions):
    """Draw graph with colored edges."""
    colors = [G[u][v]['color'] for u, v in G.edges()]
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    plt.figure(figsize=(14, 6))
    nx.draw(
        G, pos=node_positions,
        with_labels=True,
        labels=nx.get_node_attributes(G, 'label'),
        node_size=2200, node_color="#f9f9f9",
        font_size=10, font_weight='bold',
        edge_color=colors, width=weights
    )
    st.pyplot(plt.gcf())
    plt.close()


def format_tagged_output(tagged):
    """Return cleaner text format, e.g., CAT → NOUN"""
    lines = [f"{word.upper()} → {tag}" for word, tag in tagged]
    return "\n".join(lines)


# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------
st.title("HMM Part-of-Speech Tag Visualizer")

st.markdown("""
This interactive tool visualizes **Hidden Markov Model (HMM)** predictions for POS tagging.
- Green path = most likely (Viterbi) sequence  
- Gray paths = alternative tag transitions  
- Nodes represent possible tags per word  
""")

sentence_input = st.text_input("Enter a sentence:", "I don't want to do my statistics homework.")

top_n = st.slider("Number of top tags per word:", 2, 6, 3)

if st.button("Generate Visualization"):
    random.seed(hash(sentence_input))
    words = sentence_input.lower().split()

    if len(words) == 0:
        st.warning("Please enter a valid sentence.")
    else:
        G, node_positions, tagged = build_columnar_viterbi_graph(words, top_n=top_n)

        st.subheader("Tagged Output")
        st.code(format_tagged_output(tagged), language="text")

        st.subheader("Visualization of Tag Transitions")
        plot_columnar_viterbi(G, node_positions)
