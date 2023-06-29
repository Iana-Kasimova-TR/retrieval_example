import asyncio
import glob
from collections import defaultdict
from io import StringIO
from itertools import repeat
from queue import SimpleQueue

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pdfminer.converter import HTMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from sklearn.metrics import pairwise_distances
from transformers import AutoModel, AutoProcessor


def get_summarized_answer(query, df_embeds, model):
    ans1 = return_section_based_on_block_level_embedding_match_for_query(
        df_embeds, query, 2
    )
    answers1 = {i[0]: df_embeds.loc[i[0], "summary"] for i in ans1}
    return get_summary_of_answers(answers1, model, query)


async def upload_data(model):
    pdfs = [
        "example.pdf",
    ]
    # pdfs = glob.glob("./data/*.html")
    df_embeds = pd.DataFrame(columns=["page", "embeds_summary", "summary", "file_name"])
    for pdf in pdfs:
        _, slides_texts = convert_pdf_to_html_and_slide_text(pdf)

        summaries = await summarize_slides(model, slides_texts)

        text_embeds = get_embeds_for_summaries(summaries)

        data = list(text_embeds.items())
        df = pd.DataFrame(data, columns=["page", "embeds_summary"])
        text_summaries = [ans[0]["text"] for ans in summaries]
        df["summary"] = text_summaries
        df["file_name"] = pdf
    df_embeds = pd.concat([df_embeds, df], ignore_index=True)
    return df_embeds


# summaries it is index and text
def get_summary_of_answers(summaries, model, question):
    answer_text = [f"Record index: {ind}\n{ans}" for ans, ind in summaries.items()]
    joined_answers = "\n".join(answer_text)
    prompt_template = """
          Give a detailed answer on the following question:\n{question}\n
          Using excerpts provided below.
          Please provide references to record indexes ONLY in square brackets []\n
          Excerpts:\n{joined_answers}
          """
    prompt = PromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=model, prompt=prompt)
    result = chain.run({"question": question, "joined_answers": joined_answers})
    return result


def get_embeds_for_summaries(summaries):
    pages_embeds = defaultdict()
    for ans in summaries:
        pages_embeds[ans[1]] = _get_embeds(ans[0]["text"])
    return pages_embeds


def get_answer_for_one_question(model):
    template_string = "Q:Give the slide a title and summarize main points of the slide in a detailed way, preserve the structure, include all mentioned approaches and tools\nSlide:\n{slide_text}\n\nA:"
    prompt = PromptTemplate(
        input_variables=["slide_text"],
        template=template_string,
    )

    answer_chain = LLMChain(llm=model, prompt=prompt)

    return answer_chain


async def async_generate(chain, index, slide_text):
    resp = await chain.acall({"slide_text": slide_text})
    return (resp, index)


async def summarize_slides(model, slides_texts):
    chain = get_answer_for_one_question(model)
    samples = list(
        zip(
            repeat(chain),
            list(slides_texts.keys())[:5],
            list(slides_texts.values())[:5],
        )
    )

    answers = []
    answers = [async_generate(*s) for s in samples]
    answers = await asyncio.gather(*answers)
    return answers


def convert_pdf_to_html_and_slide_text(path_to_pdf: str):
    output_string = StringIO()
    with open(path_to_pdf, "rb") as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = HTMLConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
    html_content = output_string.getvalue()

    soup = BeautifulSoup(html_content, "html")
    page_dict = {}
    div_elements = soup.find_all(
        "div", style=lambda value: "position:absolute;" in value
    )
    text = []
    pages = SimpleQueue()
    for div in div_elements:
        page_num = div.find("a", attrs={"name": True})  # Extract page number
        if pages.empty() and page_num is not None:
            current_page = page_num.text.strip()
            pages.put(current_page)
            text = []
        elif page_num:
            current_page = page_num.text.strip()
            pages.put(current_page)
            page_dict[pages.get()] = " ".join(text)
            text = []
        else:
            value = div.get_text(separator=" ").strip()
            text.append(value)

    # clean text from symbols - check how display if we have step by step
    # build the dataframe with number of slides -> text from page sent to summarization -> convert summary to embed
    return html_content, page_dict


def return_section_based_on_block_level_embedding_match_for_query(
    df, query, n_sections=5
):
    all_block_embs = [emb for lst in df["embeds_summary"].values for emb in lst]
    query_emb = _get_embeds(query)
    distances = pairwise_distances(all_block_embs, query_emb, metric="cosine")
    # distance indexes will be the number of pages? as pairwise distance will the length between each embs from all_block_embs and the query_embs
    distance_indexes = np.argsort(distances, axis=0)
    return distance_indexes[:n_sections]


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def _get_embeds(txt: str):
    PRETRAINED_MODEL = "intfloat/e5-large-v2"

    processor = AutoProcessor.from_pretrained(PRETRAINED_MODEL, apply_ocr=False)
    model = AutoModel.from_pretrained(PRETRAINED_MODEL)

    encoding = processor(
        text=txt,
        return_tensors="pt",
        truncation=True,
        stride=128,
        padding="max_length",
        max_length=512,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    encoding.pop("offset_mapping")
    encoding.pop("overflow_to_sample_mapping")
    with torch.no_grad():
        model.eval()
        outputs = model(**encoding)
    # to make one vector?
    pooled_embs = average_pool(outputs.last_hidden_state, encoding["attention_mask"])
    return pooled_embs.numpy()
