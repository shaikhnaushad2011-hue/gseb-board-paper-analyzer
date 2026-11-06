import io
import json
import re
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF


# ---------------------------- PAGE SETUP ----------------------------

st.set_page_config(
    page_title="GSEB Class 10 | AI Paper Analyzer",
    page_icon="📚",
    layout="wide",
)

st.title("📚 GSEB Class 10 — AI Board Paper Analyzer")
st.caption(
    "Analyze previous-year papers for repeated questions, related concepts, "
    "chapter/topic frequency, marks and year-wise trends."
)


# ---------------------------- AI PROVIDERS ----------------------------

def call_gemini(prompt: str, api_key: str, model: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            system_instruction=(
                "You are a careful GSEB Class 10 examination-paper analyst. "
                "Return valid JSON only. Never invent missing marks or syllabus facts."
            ),
        ),
    )
    return response.text or "{}"


def call_openai(prompt: str, api_key: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful GSEB Class 10 examination-paper analyst. "
                    "Return valid JSON only. Never invent missing marks or syllabus facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or "{}"


def ask_ai(prompt: str, provider: str, api_key: str, model: str) -> dict:
    if provider == "OpenAI":
        raw = call_openai(prompt, api_key, model)
    else:
        raw = call_gemini(prompt, api_key, model)

    raw = raw.strip()
    raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)
    result = json.loads(raw)

    if not isinstance(result, dict):
        raise ValueError("The AI response was not a JSON object.")
    return result


# ---------------------------- PDF EXTRACTION ----------------------------

def extract_pdf(file_bytes: bytes) -> tuple[str, list[str]]:
    """Extract selectable text from a PDF. Scanned PDFs need OCR before upload."""
    pages = []
    warnings = []

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if len(text) < 20:
                warnings.append(
                    f"Page {page_number} has little or no selectable text. "
                    "It may be a scanned page and could not be read by this version."
                )
            pages.append(f"\n[PAGE {page_number}]\n{text}")

    return "\n".join(pages).strip(), warnings


def chunk_text(text: str, max_chars: int = 12000) -> list[str]:
    chunks = []
    current = ""

    for line in text.splitlines():
        line = line.rstrip()
        if len(current) + len(line) + 1 > max_chars and current.strip():
            chunks.append(current)
            current = line
        else:
            current += ("\n" if current else "") + line

    if current.strip():
        chunks.append(current)

    return chunks or [text]


def extract_year(filename: str):
    match = re.search(r"(20\d{2})", filename)
    return int(match.group(1)) if match else None


# ---------------------------- AI QUESTION EXTRACTION ----------------------------

def build_extraction_prompt(text: str, subject: str, syllabus: str) -> str:
    return f"""
Analyze this GSEB Class 10 {subject} question-paper text.

Extract every actual exam question, including separate sub-questions.
Do not treat instructions, headers, page numbers or answer keys as questions.
Preserve question wording as closely as possible.

Syllabus/chapter list supplied by the user:
{syllabus}

Return exactly this JSON structure:
{{
  "questions": [
    {{
      "question_number": "Q1(a)",
      "question_text": "full question text",
      "marks": 2,
      "marks_confidence": "high",
      "question_type": "short answer",
      "chapter": "chapter name from supplied syllabus, or Unclassified",
      "topic": "specific topic",
      "concept_key": "short_stable_concept_label",
      "concept_summary": "knowledge or skill being tested",
      "source_note": "page clue if visible"
    }}
  ],
  "paper_notes": ["uncertainties or extraction observations"]
}}

Rules:
- marks must be a number only if clearly printed or reliably inferable; otherwise use null.
- Never guess a chapter when unclear; use "Unclassified".
- Keep separate subparts separate when they ask different things.
- Do not duplicate questions caused by repeated text.
- Use the same concept_key for questions testing the same underlying concept,
  even if the wording or context is changed.
- Distinguish a genuinely shared concept from merely similar vocabulary.
- question_type should be one of: MCQ, very short, short answer, long answer,
  numerical, diagram, other.
- marks_confidence should be high, medium, low or unknown.
- Return valid JSON only.

PAPER TEXT:
{text}
"""


def clean_question_records(records, year, filename, subject):
    cleaned = []
    seen = set()

    for question in records:
        if not isinstance(question, dict):
            continue

        question_text = str(question.get("question_text", "")).strip()
        if len(question_text) < 5:
            continue

        dedupe_key = re.sub(r"[^a-z0-9]", "", question_text.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        raw_marks = question.get("marks")
        try:
            marks = float(raw_marks) if raw_marks is not None and str(raw_marks).strip() else None
        except (ValueError, TypeError):
            marks = None

        cleaned.append(
            {
                "Year": year if year else "Year not detected",
                "Paper": filename,
                "Subject": subject,
                "Question No.": question.get("question_number", ""),
                "Question": question_text,
                "Marks": marks,
                "Marks Confidence": question.get("marks_confidence", "unknown"),
                "Question Type": question.get("question_type", "other"),
                "Chapter": question.get("chapter", "Unclassified"),
                "Topic": question.get("topic", "Unclassified"),
                "Concept Key": str(
                    question.get("concept_key", "Unclassified")
                ).strip().lower(),
                "Concept Summary": question.get("concept_summary", ""),
                "Source Note": question.get("source_note", ""),
            }
        )

    return cleaned


# ---------------------------- COMPARISON AND REPORTS ----------------------------

def normalize_text(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def text_similarity(first, second) -> float:
    return SequenceMatcher(
        None, normalize_text(first), normalize_text(second)
    ).ratio()


def find_question_pairs(df, exact_threshold=0.96, similar_threshold=0.72):
    rows = df.to_dict("records")
    pairs = []

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            first, second = rows[i], rows[j]

            if first["Paper"] == second["Paper"]:
                continue

            score = text_similarity(first["Question"], second["Question"])
            first_concept = first.get("Concept Key", "")
            second_concept = second.get("Concept Key", "")
            same_concept = (
                first_concept not in ("", "unclassified", "none")
                and first_concept == second_concept
            )

            first_topic = first.get("Topic", "")
            second_topic = second.get("Topic", "")
            same_topic = (
                first_topic not in ("", "Unclassified")
                and first_topic == second_topic
            )

            if score >= exact_threshold:
                category = "Near-exact repeat"
            elif same_concept or (same_topic and score >= similar_threshold):
                category = "Same concept / possibly twisted"
            elif score >= similar_threshold:
                category = "Textually similar — review"
            else:
                continue

            pairs.append(
                {
                    "Category": category,
                    "Similarity %": round(score * 100, 1),
                    "Year 1": first["Year"],
                    "Question 1": first["Question"],
                    "Chapter 1": first["Chapter"],
                    "Topic 1": first["Topic"],
                    "Marks 1": first["Marks"],
                    "Year 2": second["Year"],
                    "Question 2": second["Question"],
                    "Chapter 2": second["Chapter"],
                    "Topic 2": second["Topic"],
                    "Marks 2": second["Marks"],
                    "AI Concept": (
                        first.get("Concept Summary", "") if same_concept else ""
                    ),
                }
            )

    columns = [
        "Category", "Similarity %", "Year 1", "Question 1", "Chapter 1",
        "Topic 1", "Marks 1", "Year 2", "Question 2", "Chapter 2",
        "Topic 2", "Marks 2", "AI Concept",
    ]
    return pd.DataFrame(pairs, columns=columns)


def build_reports(df):
    usable = df.copy()
    usable["Marks"] = pd.to_numeric(usable["Marks"], errors="coerce")

    topic_frequency = (
        usable.groupby(["Chapter", "Topic"], dropna=False)
        .agg(
            Questions=("Question", "count"),
            Known_Marks=("Marks", "sum"),
            Years_Appeared=("Year", "nunique"),
        )
        .reset_index()
        .sort_values(["Questions", "Known_Marks"], ascending=False)
    )

    chapter_frequency = (
        usable.groupby("Chapter", dropna=False)
        .agg(
            Questions=("Question", "count"),
            Known_Marks=("Marks", "sum"),
            Topics=("Topic", "nunique"),
            Years_Appeared=("Year", "nunique"),
        )
        .reset_index()
    )

    chapter_frequency["Question Share %"] = (
        chapter_frequency["Questions"] / max(len(usable), 1) * 100
    ).round(1)

    chapter_frequency = chapter_frequency.sort_values(
        ["Questions", "Known_Marks"], ascending=False
    ).reset_index(drop=True)
    chapter_frequency.insert(
        0, "Frequency Rank", range(1, len(chapter_frequency) + 1)
    )

    year_chapter = pd.crosstab(
        usable["Year"], usable["Chapter"]
    ).reset_index()

    year_summary = (
        usable.groupby("Year")
        .agg(
            Questions=("Question", "count"),
            Known_Marks=("Marks", "sum"),
            Questions_With_Known_Marks=("Marks", "count"),
        )
        .reset_index()
    )

    question_types = (
        usable.groupby("Question Type")
        .agg(
            Questions=("Question", "count"),
            Known_Marks=("Marks", "sum"),
        )
        .reset_index()
        .sort_values("Questions", ascending=False)
    )

    return topic_frequency, chapter_frequency, year_chapter, year_summary, question_types


# ---------------------------- SIDEBAR SETTINGS ----------------------------

DEFAULT_SYLLABUS = {
    "Science": """Chemical Reactions and Equations
Acids, Bases and Salts
Metals and Non-metals
Carbon and its Compounds
Life Processes
Control and Coordination
How do Organisms Reproduce?
Heredity
Light – Reflection and Refraction
The Human Eye and the Colourful World
Electricity
Magnetic Effects of Electric Current
Our Environment""",
    "Mathematics": """Real Numbers
Polynomials
Pair of Linear Equations in Two Variables
Quadratic Equations
Arithmetic Progressions
Triangles
Coordinate Geometry
Introduction to Trigonometry
Some Applications of Trigonometry
Circles
Areas Related to Circles
Surface Areas and Volumes
Statistics
Probability""",
    "English": """Reading Comprehension
Writing
Grammar
Prose
Poetry""",
    "Social Science": """History
Geography
Political Science
Economics""",
    "Gujarati": """Reading
Writing
Grammar
Prose
Poetry""",
    "Hindi": """Reading
Writing
Grammar
Prose
Poetry""",
}

with st.sidebar:
    st.header("Analysis settings")

    subject_choice = st.selectbox(
        "Subject",
        [
            "Science", "Mathematics", "English", "Social Science",
            "Gujarati", "Hindi", "Other",
        ],
    )
    subject = subject_choice
    if subject_choice == "Other":
        subject = st.text_input("Enter subject name", "Other")

    provider = st.radio("AI provider", ["Gemini", "OpenAI"])

    if provider == "Gemini":
        api_key = st.text_input(
            "Gemini API key", type="password",
            help="Your key is used for this session and is not displayed.",
        )
        model_name = st.text_input("Gemini model", "gemini-2.5-flash")
        st.caption("A Google AI Studio API key is required. Free-tier limits may apply.")
    else:
        api_key = st.text_input("OpenAI API key", type="password")
        model_name = st.text_input("OpenAI model", "gpt-4.1-mini")
        st.caption("OpenAI API usage may incur charges.")

    st.divider()
    st.subheader("GSEB Class 10 syllabus")
    syllabus = st.text_area(
        "One chapter/topic per line",
        DEFAULT_SYLLABUS.get(subject, "Enter syllabus topics here"),
        height=230,
    )


# ---------------------------- MAIN UPLOAD AND ANALYSIS ----------------------------

st.info(
    "Upload text-based PDF question papers. Include the year in each filename, "
    "for example GSEB_Science_2024.pdf. Scanned/image-only PDFs are not supported "
    "by this hosted version unless OCR is added separately."
)

uploads = st.file_uploader(
    "Upload previous-year question papers",
    type=["pdf"],
    accept_multiple_files=True,
)

with st.expander("How the analysis works"):
    st.markdown(
        """
        1. Extract selectable text from each PDF.
        2. Ask the selected AI model to identify questions, marks, chapters, topics and concepts.
        3. Compare question wording and AI-assigned concepts to flag likely repeats and twisted questions.
        4. Calculate chapter/topic frequency, known marks and year-wise trends.
        5. Export the results to an Excel workbook.

        **Important:** This is historical analysis, not a guaranteed prediction of the next exam.
        Verify extracted questions, marks and chapter assignments against the original papers.
        """
    )

if st.button("🔍 Analyze uploaded papers", type="primary", use_container_width=True):
    if not uploads:
        st.error("Please upload at least one PDF question paper.")
        st.stop()

    if not api_key.strip():
        st.error("Enter your selected AI provider's API key in the sidebar.")
        st.stop()

    if not model_name.strip():
        st.error("Enter a model name in the sidebar.")
        st.stop()

    if not syllabus.strip():
        st.error("Enter the subject's syllabus chapters/topics.")
        st.stop()

    all_records = []
    notes = []
    progress = st.progress(0)
    status = st.empty()

    try:
        for file_index, uploaded_file in enumerate(uploads):
            status.write(f"Reading {uploaded_file.name}…")
            pdf_text, pdf_warnings = extract_pdf(uploaded_file.getvalue())
            notes.extend([f"{uploaded_file.name}: {note}" for note in pdf_warnings])

            if not pdf_text.strip():
                notes.append(
                    f"{uploaded_file.name}: no selectable text found. "
                    "This may be a scanned PDF; use a text-based PDF or add OCR support."
                )
                progress.progress((file_index + 1) / len(uploads))
                continue

            chunks = chunk_text(pdf_text)
            extracted_questions = []

            for chunk_index, chunk in enumerate(chunks):
                status.write(
                    f"AI is analyzing {uploaded_file.name} "
                    f"({chunk_index + 1}/{len(chunks)})…"
                )
                prompt = build_extraction_prompt(chunk, subject, syllabus)
                ai_result = ask_ai(prompt, provider, api_key, model_name)

                questions = ai_result.get("questions", [])
                if isinstance(questions, list):
                    extracted_questions.extend(questions)

                paper_notes = ai_result.get("paper_notes", [])
                if isinstance(paper_notes, list):
                    notes.extend(
                        [f"{uploaded_file.name}: {note}" for note in paper_notes]
                    )

            all_records.extend(
                clean_question_records(
                    extracted_questions,
                    extract_year(uploaded_file.name),
                    uploaded_file.name,
                    subject,
                )
            )
            progress.progress((file_index + 1) / len(uploads))

        if not all_records:
            st.error(
                "No questions were extracted. Check that your PDFs contain selectable text "
                "and that the API key/model are correct."
            )
            if notes:
                with st.expander("Details"):
                    for note in notes:
                        st.write("• " + str(note))
            st.stop()

        questions_df = pd.DataFrame(all_records)
        pairs_df = find_question_pairs(questions_df)
        (
            topic_frequency,
            chapter_frequency,
            year_chapter,
            year_summary,
            question_types,
        ) = build_reports(questions_df)

        st.session_state["analysis_result"] = {
            "questions": questions_df,
            "pairs": pairs_df,
            "topic_frequency": topic_frequency,
            "chapter_frequency": chapter_frequency,
            "year_chapter": year_chapter,
            "year_summary": year_summary,
            "question_types": question_types,
            "notes": notes,
        }

        status.success("Analysis completed. Review the results below.")

    except Exception as error:
        st.error(f"Analysis stopped: {error}")
        st.info(
            "Check your API key, internet connection, model name and API quota. "
            "If the error mentions a package, check requirements.txt."
        )


# ---------------------------- RESULTS DASHBOARD ----------------------------

if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]
    questions_df = result["questions"]
    pairs_df = result["pairs"]
    topic_frequency = result["topic_frequency"]
    chapter_frequency = result["chapter_frequency"]
    year_chapter = result["year_chapter"]
    year_summary = result["year_summary"]
    question_types = result["question_types"]

    st.divider()
    st.header("📊 Analysis dashboard")

    known_marks = pd.to_numeric(questions_df["Marks"], errors="coerce")
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Papers analyzed", questions_df["Paper"].nunique())
    metric2.metric("Questions extracted", len(questions_df))
    metric3.metric("Known marks total", f"{known_marks.sum():g}")
    metric4.metric("Repeat/concept pairs", len(pairs_df))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Chapter ranking",
            "Topic frequency",
            "Repeated & twisted",
            "Marks analysis",
            "Year-wise trends",
            "All questions",
        ]
    )

    with tab1:
        st.subheader("Chapter ranking by historical frequency")
        st.caption(
            "This ranking describes only the papers you uploaded; it does not predict "
            "or guarantee future exam weightage."
        )
        st.dataframe(chapter_frequency, use_container_width=True, hide_index=True)
        if not chapter_frequency.empty:
            st.bar_chart(chapter_frequency.set_index("Chapter")[["Questions"]])

    with tab2:
        st.subheader("Most and least frequently asked topics")
        st.dataframe(topic_frequency, use_container_width=True, hide_index=True)
        if not topic_frequency.empty:
            st.bar_chart(topic_frequency.set_index("Topic")[["Questions"]])

    with tab3:
        st.subheader("Repeated and twisted question candidates")
        st.caption(
            "Near-exact repeats use wording similarity. Possible twisted questions use "
            "the AI's concept labels. Review each pair against the original papers."
        )
        if pairs_df.empty:
            st.info("No likely repeated or concept-related pairs were found.")
        else:
            categories = sorted(pairs_df["Category"].dropna().unique().tolist())
            selected_categories = st.multiselect(
                "Filter categories",
                categories,
                default=categories,
            )
            st.dataframe(
                pairs_df[pairs_df["Category"].isin(selected_categories)],
                use_container_width=True,
                hide_index=True,
            )

    with tab4:
        st.subheader("Marks analysis")
        st.caption("Only marks identified by the AI are included in known-mark totals.")
        st.metric("Questions with known marks", int(known_marks.notna().sum()))
        st.metric("Questions with missing marks", int(known_marks.isna().sum()))
        st.dataframe(
            chapter_frequency[
                ["Chapter", "Questions", "Known_Marks", "Years_Appeared"]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Question format distribution")
        st.dataframe(question_types, use_container_width=True, hide_index=True)
        if not question_types.empty:
            st.bar_chart(question_types.set_index("Question Type")[["Questions"]])

    with tab5:
        st.subheader("Chapter frequency by year")
        st.dataframe(year_chapter, use_container_width=True, hide_index=True)
        st.subheader("Year-wise paper summary")
        st.dataframe(year_summary, use_container_width=True, hide_index=True)
        st.caption(
            "Compare equivalent exam types and syllabus versions. Different paper formats "
            "or syllabus changes can affect historical frequencies."
        )

    with tab6:
        st.subheader("Extracted question database")
        st.dataframe(questions_df, use_container_width=True, hide_index=True)

    if result["notes"]:
        with st.expander("Extraction notes and uncertainties"):
            for note in result["notes"]:
                st.write("• " + str(note))

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        questions_df.to_excel(writer, sheet_name="All Questions", index=False)
        chapter_frequency.to_excel(writer, sheet_name="Chapter Ranking", index=False)
        topic_frequency.to_excel(writer, sheet_name="Topic Frequency", index=False)
        pairs_df.to_excel(writer, sheet_name="Repeated Twisted", index=False)
        year_chapter.to_excel(writer, sheet_name="Year Trends", index=False)
        year_summary.to_excel(writer, sheet_name="Year Summary", index=False)
        question_types.to_excel(writer, sheet_name="Question Types", index=False)

    st.download_button(
        "⬇️ Download complete Excel analysis",
        data=excel_buffer.getvalue(),
        file_name="GSEB_Class10_AI_Paper_Analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.warning(
        "AI extraction and classification can contain errors. Verify question wording, "
        "marks and chapter assignments against the original papers. Historical frequency "
        "cannot guarantee what will appear in a future examination."
    )
else:
    st.markdown("### Your analysis will appear here")
    st.write(
        "Upload previous-year question papers to begin. For best results, use complete, "
        "text-based PDFs and include the exam year in each filename."
    )
