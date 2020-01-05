import streamlit as st

# NPL pakages
import spacy
from textblob import TextBlob
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# Function to Analyse Tokens and Lemas (spaCy)

def text_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	tokens = [token.text for token in docx]
	allData = [('"Token": {}, "Lemma": {} '.format(token.text, token.lemma_)) for token in docx]
	return allData


# Function to Extract Named Entities (spaCy)	

def entity_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	tokens = [token.text for token in docx]
	entities =  [(entity.text, entity.label_) for entity in docx.ents]
	allData = [('"Token": {}, "Entity": {} \n'.format(tokens, entities))]
	return allData


# Function for Sumy Summarization

def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx, Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document, 3)
	summary_list =  [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


# Main function

def main():
	'''NLP App with Streamlit'''
	st.title("NLPify with Streamlit")
	st.subheader("Natural Language Processing online")

	### NLP1: TOKENIZATION ###
	if st.checkbox("Show Tokens and Lemmas"):
		st.subheader("Tokenize your text")
		message = st.text_area("Enter your text", "Type or copy here...", '1') 
		# '1' => key (str) – An optional string to use as the unique key for the widget. 
		if st.button("Analyse", '1'):
			nlp_result = text_analyzer(message)
			#st.success(nlp_result) # caixa de texto verde
			st.json(nlp_result) # formato json



	### NLP2: NER ###
	if st.checkbox("Show Named Entities"):
		st.subheader("Extract Named Entities form your text")
		message = st.text_area("Enter your text", "Type or copy here...", '2') 
		if st.button("Extract", '2'):
			nlp_result = entity_analyzer(message)
			#st.success(nlp_result) # caixa de texto verde
			st.json(nlp_result) # formato json


	### NLP3: SENTIMENTS ANALYSIS ###
	if st.checkbox("Show Sentiment Analysis"):
		st.subheader("Analyse sentiments of your text")
		message = st.text_area("Enter your text", "Type or copy here...", '3')
		if st.button("Analyse", '3'):
			blob = TextBlob(message)
			result_sentiment = blob.sentiment
			st.success(result_sentiment)


	### NLP4: TEXT SUMMERIZATION ###
	if st.checkbox("Show Text summarization"):
		st.subheader("Summarize your text")
		message = st.text_area("Enter your text", "Type or copy here...", '4')
		summary_options = st.selectbox("Choise your summarizer", ("gensim", "sumy"))
		if st.button("Summarize", '4'):
			if summary_options == "gensim":
				st.text("Using Gensim ...")
				summary_result = summarize(message)
			elif summary_options == "sumy":
				st.text("Using Sumy ...")
				summary_result = sumy_summarizer(message)
			else:
				st.warning("Using Default Summarizer")
				st.text("Using Gensim ...")
				summary_result = summarize(message)

			st.success(summary_result)

	### Sidebar
	st.sidebar.subheader("About the App")
	st.sidebar.text("NLPify App with Streamlit")
	st.sidebar.info("Credits to the Streamlit team")

if __name__ == "__main__":
	main()