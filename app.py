import os
import json
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI


def set_api_key_from_ui():
    # Allow users to optionally provide an API key via the UI without storing it
    with st.sidebar:
        st.header("API Settings")
        user_key = st.text_input("OpenAI API Key", type="password", help="Your key is only used in this session.")
        if user_key:
            os.environ["OPENAI_API_KEY"] = user_key


def get_client() -> OpenAI:
    return OpenAI()


def build_user_prompt(
    profile: Dict[str, Any],
    prefs: Dict[str, Any],
    provided_companies: List[str],
    top_n: int,
) -> str:
    prompt = f"""
You are a startup-curation research agent.

Task:
Curate a list of high-potential startups to apply to, focusing on:
- Core products and what problems they solve
- Founders’ mentality and culture signals
- Alignment with the candidate’s interests and skills
- Bonus points for open-source involvement (products, tooling, or contributions)

Input:
- Candidate Profile:
  - Name: {profile.get("name") or "N/A"}
  - Headline/Role: {profile.get("headline") or "N/A"}
  - Skills/Tech: {profile.get("skills") or "N/A"}
  - Interests/Theses: {profile.get("interests") or "N/A"}
  - Values/Culture: {profile.get("values") or "N/A"}
  - Notable Links (GitHub/Portfolio/LinkedIn): {profile.get("links") or "N/A"}

- Preferences/Filters:
  - Sectors/Problem Areas: {prefs.get("sectors") or "Any"}
  - Stage Preference: {prefs.get("stage") or "Any"}
  - Team Size Range: {prefs.get("team_size") or "Any"}
  - Location/Remote: {prefs.get("location") or "Any"}
  - Open-Source Importance: {prefs.get("oss_importance") or "Neutral"}
  - Other Constraints: {prefs.get("other") or "None"}
  - Exclude Companies: {prefs.get("exclude") or "None"}
  - Geographic Focus: {prefs.get("geo") or "Any"}

- Provided companies to vet (optional):
  {provided_companies if provided_companies else "None"}

Output requirements:
- Return ONLY valid, minified JSON (no comments, no markdown fences).
- If you are unsure of a fact, use "unknown" or empty strings.
- Do NOT invent links. Only include links you are reasonably confident about.
- Prefer startups that meaningfully align with the candidate profile.
- Include both: (1) fresh/newer companies if relevant, and (2) enduring, mission-aligned companies.
- If the user provided companies, prioritize vetting and ranking them, and fill remaining slots with additional suggestions.

JSON schema:
{{
  "generated_at": "<ISO-8601 timestamp>",
  "query_summary": "<short summary of how you interpreted the candidate's profile and constraints>",
  "startups": [
    {{
      "name": "<company name>",
      "website": "<official website or empty>",
      "hq_location": "<city/country or remote or unknown>",
      "stage": "<pre-seed|seed|series-a|series-b|growth|unknown>",
      "team_size": "<approx or unknown>",
      "core_product": "<clear summary of product and problem solved>",
      "founders": [
        {{
          "name": "<founder name or unknown>",
          "background": "<short background or unknown>",
          "mentality_notes": "<signals on builder mindset, product-first, frugality, research orientation, etc.>"
        }}
      ],
      "open_source_involvement": {{
        "level": "<none|partial|core|unknown>",
        "repos": [
          {{"name": "<repo name>", "url": "<repo url>"}}
        ]
      }},
      "why_aligned": "<explicit alignment with candidate's interests, skills, and values>",
      "suggested_roles": ["<role 1>", "<role 2>"],
      "example_outreach": "<a short, tailored outreach note the candidate could send>",
      "sources": [
        {{"label": "website", "url": "<url or empty>"}},
        {{"label": "github", "url": "<url or empty>"}},
        {{"label": "other", "url": "<url or empty>"}}
      ],
      "confidence": <float between 0 and 1>
    }}
  ],
  "notes": "<disclaimers and what to validate next>",
  "next_actions": ["<suggested next steps for the candidate>"]
}}

Constraints:
- Limit the list to exactly {top_n} startups.
- Ensure diversity of choices while adhering to preferences.
- Keep fields concise but specific.
"""
    return prompt.strip()


def clean_json_text(text: str) -> str:
    text = text.strip()
    # Remove Markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1])
    return text.strip()


def call_model(client: OpenAI, model: str, prompt: str, temperature: float = 0.4, max_tokens: int = 1800) -> str:
    response = client.chat.completions.create(
        model=model,  # "gpt-4" or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    try:
        cleaned = clean_json_text(text)
        return json.loads(cleaned)
    except Exception:
        return None


def render_startup_card(s: Dict[str, Any]):
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        title = s.get("name") or "Unnamed Startup"
        website = (s.get("website") or "").strip()
        if website:
            st.subheader(f"{title} — {website}")
        else:
            st.subheader(title)
        st.caption(s.get("hq_location") or "Location: unknown")

    with col2:
        stage = s.get("stage") or "unknown"
        size = s.get("team_size") or "unknown"
        st.metric("Stage", stage)
        st.metric("Team Size", size)

    st.write(f"Core Product: {s.get('core_product') or 'unknown'}")

    founders = s.get("founders") or []
    if founders:
        with st.expander("Founders and mentality", expanded=False):
            for f in founders:
                name = f.get("name") or "unknown"
                bg = f.get("background") or "unknown"
                notes = f.get("mentality_notes") or "—"
                st.write(f"- {name}: {bg}")
                if notes and notes != "—":
                    st.caption(f"  Mentality: {notes}")

    oss = s.get("open_source_involvement") or {}
    oss_level = oss.get("level") or "unknown"
    repos = oss.get("repos") or []
    with st.expander(f"Open Source: {oss_level}", expanded=False):
        if repos:
            for r in repos:
                rname = r.get("name") or "repo"
                rurl = r.get("url") or ""
                if rurl:
                    st.write(f"- {rname}: {rurl}")
                else:
                    st.write(f"- {rname}")
        else:
            st.write("No repos listed.")

    st.write(f"Why aligned: {s.get('why_aligned') or '—'}")

    roles = s.get("suggested_roles") or []
    if roles:
        st.write("Suggested roles: " + ", ".join(roles))

    outreach = s.get("example_outreach") or ""
    if outreach:
        with st.expander("Example outreach", expanded=False):
            st.code(outreach)

    sources = s.get("sources") or []
    if sources:
        with st.expander("Sources / Links", expanded=False):
            for src in sources:
                label = src.get("label") or "link"
                url = src.get("url") or ""
                if url:
                    st.write(f"- {label}: {url}")
                else:
                    st.write(f"- {label}: (unknown)")

    conf = s.get("confidence")
    if isinstance(conf, (float, int)):
        st.progress(min(max(conf, 0.0), 1.0))


def main():
    st.set_page_config(page_title="Startup Curator Agent", page_icon="🚀", layout="wide")
    st.title("🚀 Startup Curator Agent")
    st.write("Get a curated list of startups to apply to — grounded in core products, founder mentality, and alignment with your interests. Bonus preference for open-source friendly companies.")

    set_api_key_from_ui()

    with st.sidebar:
        st.header("Your Profile")
        name = st.text_input("Name (optional)")
        headline = st.text_input("Headline / Role (e.g., ML Engineer, Product Engineer)")
        skills = st.text_area("Skills / Tech Stack", placeholder="e.g., Python, PyTorch, Go, Rust, React, Postgres, Kubernetes")
        interests = st.text_area("Interests / Theses", placeholder="What domains excite you? e.g., OSS infra, devtools, AI agents, bio, climate, fintech")
        values = st.text_area("Values / Culture Preferences", placeholder="e.g., product-first, frugal, fast iteration, founder-led")
        links = st.text_area("Links (GitHub, Portfolio, LinkedIn)", placeholder="https://github.com/..., https://www.linkedin.com/in/...")

        st.header("Preferences")
        sectors = st.text_input("Target Sectors/Problem Areas", placeholder="e.g., DevTools, AI Infrastructure, Data, Security")
        stage = st.selectbox("Stage Preference", ["Any", "pre-seed", "seed", "series-a", "series-b", "growth"])
        team_size = st.text_input("Team Size Range", placeholder="e.g., 5-50")
        location = st.text_input("Location / Remote", placeholder="e.g., Remote-first; SF Bay Area; EU-friendly")
        oss_importance = st.select_slider("Open-Source Importance", options=["Low", "Neutral", "High"], value="High")
        geo = st.text_input("Geographic Focus", placeholder="Any")

        other = st.text_area("Other Constraints", placeholder="Visa needs, funding preferences, industries to avoid, etc.")
        exclude = st.text_area("Exclude Companies", placeholder="Comma-separated names to exclude")

        st.header("Vetting list (optional)")
        provided_raw = st.text_area("Provide companies to vet (one per line)", placeholder="company1\ncompany2\n...")

        st.header("Generation Settings")
        model_choice = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"], index=0)
        top_n = st.slider("How many startups?", min_value=3, max_value=15, value=8, step=1)
        temperature = st.slider("Creativity (temperature)", min_value=0.0, max_value=1.0, value=0.4, step=0.1)

    client = get_client()

    profile = {
        "name": name.strip() if name else "",
        "headline": headline.strip() if headline else "",
        "skills": skills.strip() if skills else "",
        "interests": interests.strip() if interests else "",
        "values": values.strip() if values else "",
        "links": links.strip() if links else "",
    }

    prefs = {
        "sectors": sectors.strip() if sectors else "",
        "stage": stage,
        "team_size": team_size.strip() if team_size else "",
        "location": location.strip() if location else "",
        "oss_importance": oss_importance,
        "geo": geo.strip() if geo else "",
        "other": other.strip() if other else "",
        "exclude": exclude.strip() if exclude else "",
    }

    provided_companies = []
    if provided_raw:
        provided_companies = [line.strip() for line in provided_raw.splitlines() if line.strip()]

    st.markdown("---")
    st.subheader("Your Inputs Summary")
    with st.expander("View summary", expanded=False):
        st.write("Profile:", profile)
        st.write("Preferences:", prefs)
        if provided_companies:
            st.write("Provided companies to vet:", provided_companies)

    generate = st.button("Generate Curated List", type="primary")

    if generate:
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("Please provide your OpenAI API key in the sidebar.")
            st.stop()

        with st.spinner("Curating startups..."):
            prompt = build_user_prompt(profile, prefs, provided_companies, top_n)
            try:
                raw_text = call_model(client, model_choice, prompt, temperature=temperature, max_tokens=2200)
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
                st.stop()

        data = parse_json_response(raw_text)
        if not data:
            st.error("Failed to parse the model response as JSON. Showing raw output below.")
            with st.expander("Raw model output"):
                st.text(raw_text)
            st.stop()

        st.success("Curated list ready!")
        st.caption(data.get("query_summary") or "")

        startups = data.get("startups") or []
        for s in startups:
            st.markdown("---")
            render_startup_card(s)

        st.markdown("---")
        if data.get("notes"):
            st.info(data["notes"])
        if data.get("next_actions"):
            st.write("Next actions:")
            for na in data["next_actions"]:
                st.write(f"- {na}")

        st.download_button(
            label="Download JSON",
            data=json.dumps(data, indent=2),
            file_name="startup_curation.json",
            mime="application/json",
        )

    st.markdown("---")
    st.caption("Note: This tool uses language models and may have inaccuracies. Validate important details directly from company sources before applying.")


if __name__ == "__main__":
    main()