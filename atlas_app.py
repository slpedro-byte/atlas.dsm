import streamlit as st
import numpy as np

st.set_page_config(
    page_title="ATLAS - Decision Support Model",
    page_icon="A",
    layout="centered"
)

WEIGHTS = {
    "data_availability": 0.300,
    "problem_scope":     0.300,
    "org_maturity":      0.200,
    "digital_maturity":  0.100,
    "time_available":    0.100,
}

FRAMEWORK_SCORES = {
    "DMAIC": {"data_availability":5,"problem_scope":4,"org_maturity":4,"digital_maturity":3,"time_available":2},
    "A3":    {"data_availability":3,"problem_scope":5,"org_maturity":2,"digital_maturity":2,"time_available":4},
    "Lean":  {"data_availability":2,"problem_scope":2,"org_maturity":3,"digital_maturity":3,"time_available":3},
    "PDCA":  {"data_availability":2,"problem_scope":4,"org_maturity":1,"digital_maturity":1,"time_available":5},
}

FRAMEWORK_INFO = {
    "DMAIC": {
        "description": "Six Sigma methodology for reducing measurable variability in existing processes through a rigorous, data-driven 5-phase cycle.",
        "when": "Complex quality problems with historical data, 3 to 6 months available, Green or Black Belt expertise.",
        "tools": "SIPOC, MSA, Fishbone, DOE, Control Charts, SPC"
    },
    "A3": {
        "description": "Structured one-page problem-solving format for well-scoped, bounded problems requiring root cause analysis.",
        "when": "Problem is already well-defined, low-to-medium maturity, 1 to 4 weeks available.",
        "tools": "A3 Report, 5 Whys, Fishbone, PDCA logic"
    },
    "Lean": {
        "description": "System-level philosophy targeting waste elimination and flow optimisation across the value stream.",
        "when": "Flow and waste problems across a process or value stream, identifiable non-value-added activities.",
        "tools": "VSM (Value Stream Mapping), Kaizen Events, Kanban, 5S"
    },
    "PDCA": {
        "description": "Lightweight iterative improvement cycle for incremental, hypothesis-driven problems.",
        "when": "Simple incremental improvements, low maturity organisations, results needed in days or weeks.",
        "tools": "Plan-Do-Check-Act cycle, Run Charts, Control Charts"
    },
    "8D": {
        "description": "Reactive methodology with immediate customer containment before root cause investigation.",
        "when": "Customer complaint active, production stopped, regulatory pressure. Urgency of containment required.",
        "tools": "D0 to D8 steps, Containment Action, Fishbone, 5 Whys, Mistake-Proofing"
    },
    "TRIZ": {
        "description": "Innovation-oriented framework for resolving technical contradictions in engineering problems.",
        "when": "New design challenge with a technical contradiction. Improving one parameter degrades another.",
        "tools": "Contradiction Matrix, 40 Inventive Principles, ARIZ"
    },
    "DMADV": {
        "description": "Design for Six Sigma. Builds quality into new products or processes from the outset.",
        "when": "New product or process design, no existing process to improve, 4 to 9 months available.",
        "tools": "VOC, QFD/House of Quality, DFMEA, Design Verification"
    },
}

HYBRID_MAP = {
    ("DMAIC","A3"): (
        "A3 + DMAIC",
        """
**How to apply this hybrid:**

Begin with an A3 to scope and structure the problem. The A3 provides a disciplined one-page format to define the problem statement, map the current situation, and identify the root cause using 5 Whys or fishbone analysis. If root cause investigation reveals multi-variable statistical interactions that exceed A3 analytical tools, escalate to DMAIC for the Analyse and Improve phases.

**Sequence:** A3 (Define, Current State, Root Cause) -> DMAIC (Analyse, Improve, Control)

**When to escalate:** When root cause analysis at D4 equivalent in A3 reveals process variability requiring statistical methods (regression, hypothesis testing, DOE).

**Literature basis:** Kumar Phanden et al. (2022); Cantini et al. (2024)
"""
    ),
    ("A3","DMAIC"): (
        "A3 + DMAIC",
        """
**How to apply this hybrid:**

Begin with an A3 to scope and structure the problem. The A3 provides a disciplined one-page format to define the problem statement, map the current situation, and identify the root cause using 5 Whys or fishbone analysis. If root cause investigation reveals multi-variable statistical interactions that exceed A3 analytical tools, escalate to DMAIC for the Analyse and Improve phases.

**Sequence:** A3 (Define, Current State, Root Cause) -> DMAIC (Analyse, Improve, Control)

**When to escalate:** When root cause analysis at D4 equivalent in A3 reveals process variability requiring statistical methods (regression, hypothesis testing, DOE).

**Literature basis:** Kumar Phanden et al. (2022); Cantini et al. (2024)
"""
    ),
    ("Lean","PDCA"): (
        "Lean + PDCA",
        """
**How to apply this hybrid:**

Deploy PDCA cycles first to build continuous improvement culture and generate early wins through fast, low-overhead improvement iterations. In parallel, initiate Value Stream Mapping (VSM) to diagnose waste and flow inefficiencies at the system level. As organisational maturity grows, Kaizen Events address specific waste types identified in the VSM, with each Kaizen cycle structured as a PDCA iteration.

**Sequence:** PDCA (rapid cycles, culture building) + VSM (system diagnosis) -> Kaizen Events (targeted improvement, each structured as PDCA)

**When to use:** Organisations at early CI maturity where full Lean deployment is not yet feasible, but management commitment is growing.

**Literature basis:** Arredondo-Soto et al. (2021); Ishak et al. (2019)
"""
    ),
    ("PDCA","Lean"): (
        "Lean + PDCA",
        """
**How to apply this hybrid:**

Deploy PDCA cycles first to build continuous improvement culture and generate early wins through fast, low-overhead improvement iterations. In parallel, initiate Value Stream Mapping (VSM) to diagnose waste and flow inefficiencies at the system level. As organisational maturity grows, Kaizen Events address specific waste types identified in the VSM, with each Kaizen cycle structured as a PDCA iteration.

**Sequence:** PDCA (rapid cycles, culture building) + VSM (system diagnosis) -> Kaizen Events (targeted improvement, each structured as PDCA)

**When to use:** Organisations at early CI maturity where full Lean deployment is not yet feasible, but management commitment is growing.

**Literature basis:** Arredondo-Soto et al. (2021); Ishak et al. (2019)
"""
    ),
}

def run_topsis(candidates, context):
    criteria = list(WEIGHTS.keys())
    n = len(candidates)
    matrix = np.zeros((n, len(criteria)))
    for i, fw in enumerate(candidates):
        for j, cr in enumerate(criteria):
            matrix[i, j] = FRAMEWORK_SCORES[fw][cr] * context[cr] / 5
    norm_matrix = matrix / 5.0 
    w = np.array([WEIGHTS[c] for c in criteria])
    weighted = norm_matrix * w
    pis = weighted.max(axis=0)
    nis = weighted.min(axis=0)
    d_plus  = np.sqrt(((weighted - pis)**2).sum(axis=1))
    d_minus = np.sqrt(((weighted - nis)**2).sum(axis=1))
    ci = d_minus / (d_plus + d_minus + 1e-10)
    return sorted(zip(candidates, ci), key=lambda x: x[1], reverse=True)

def show_direct(fw):
    st.divider()
    st.markdown("### Recommendation")
    if fw == "8D":
        st.error(f"**{fw}**")
    else:
        st.success(f"**{fw}**")
    st.markdown("*Hard-constraint recommendation. No ranking needed.*")
    info = FRAMEWORK_INFO[fw]
    with st.expander("What is this framework?", expanded=True):
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Best when:** {info['when']}")
        st.markdown(f"**Key tools:** {info['tools']}")
    if st.button("Start again"):
        st.session_state.clear()
        st.rerun()

def show_topsis_stage(candidates):
    st.divider()
    st.markdown("### Stage 2 - TOPSIS Ranking")
    st.caption("Rate your organisation's context. The model will calculate the best fit.")

    col1, col2 = st.columns(2)
    with col1:
        data     = st.slider("Data availability\n\n*Historical process data exists?*", 1, 5, 3, help="1 = none, 5 = rich dataset")
        scope    = st.slider("Problem scope\n\n*How well-defined is the problem?*", 1, 5, 3, help="1 = vague, 5 = very clear")
        maturity = st.slider("Organisational maturity\n\n*CI skills and culture?*", 1, 5, 3, help="1 = no CI, 3 = some projects with experts, 5 = dedicated team")
    with col2:
        digital  = st.slider("Digital maturity\n\n*IoT, dashboards, analytics?*", 1, 5, 3, help="1 = paper-based, 5 = advanced analytics")
        time     = st.slider("Time available\n\n*How long to solve this?*", 1, 5, 3, help="1 = days, 3 = weeks, 5 = 6 or more months")

    context = {
        "data_availability": data,
        "problem_scope":     scope,
        "org_maturity":      maturity,
        "digital_maturity":  digital,
        "time_available":    time,
    }

    if st.button("Run ATLAS", type="primary", use_container_width=True):
        ranked = run_topsis(candidates, context)
        top_fw, top_ci = ranked[0]
        second_fw, second_ci = ranked[1]
        gap = top_ci - second_ci

        st.divider()
        st.markdown("### TOPSIS Ranking")
        cols = st.columns(len(ranked))
        for i, (fw, ci) in enumerate(ranked):
            with cols[i]:
                rank_label = "1st" if i == 0 else "2nd"
                st.metric(
                    label=f"{rank_label} - {fw}",
                    value=f"Ci = {ci:.3f}",
                    delta="Top choice" if i == 0 else f"Gap: {gap:.3f}"
                )

        st.divider()
        st.markdown("### Recommendation")

        hybrid_key = (top_fw, second_fw)
        if gap < 0.15 and top_ci > 0.40 and second_ci > 0.40 and hybrid_key in HYBRID_MAP:
            hybrid_name, hybrid_desc = HYBRID_MAP[hybrid_key]
            st.warning(f"**Hybrid Recommended: {hybrid_name}**")
            st.markdown(f"*Gap = {gap:.3f}. Both frameworks are strong candidates. A hybrid approach may be superior.*")
            st.markdown(hybrid_desc)
        else:
            st.success(f"**{top_fw}** (Ci = {top_ci:.3f})")

        info = FRAMEWORK_INFO[top_fw]
        with st.expander(f"About {top_fw}", expanded=True):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Best when:** {info['when']}")
            st.markdown(f"**Key tools:** {info['tools']}")

        if top_fw == "Lean":
            with st.expander("Lean tool guidance", expanded=True):
                st.markdown("""
**Within Lean Manufacturing, the recommended tools are:**

- **VSM (Value Stream Mapping):** map and diagnose waste and flow inefficiencies at the system level before any intervention begins (Yang et al., 2025).
- **Kaizen Events:** structured rapid improvement workshops targeting specific waste types identified during VSM (Rossini et al., 2021).
- **Kanban:** implement pull-based flow control to stabilise and sustain improvements after Kaizen Events (Ishak et al., 2019).

Tool selection follows from the specific waste type identified during the VSM analysis.
""")

        if st.button("Start again"):
            st.session_state.clear()
            st.rerun()

# MAIN APP

st.markdown("""
<div style='text-align:center; padding:1.5rem 0 0.5rem 0;'>
    <h1 style='font-size:2.2rem; margin-bottom:0;'>ATLAS</h1>
    <p style='color:#888; font-size:1rem; margin-top:0.2rem;'>
        Adaptive Tool Leading problem-solving Analysis and Selection
    </p>
    <p style='color:#aaa; font-size:0.85rem;'>
        Decision Support Model - NOVA School of Science and Technology
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()
st.markdown("### Stage 1 - Decision Tree")
st.caption("Three binary questions to filter structurally incompatible frameworks.")

q1 = st.radio(
    "**Q1 - Does a process or product already exist?**",
    ["Select an answer...",
     "Yes - I want to improve something that already exists",
     "No - I am designing something new"],
    key="q1"
)

if q1 != "Select an answer...":

    if "No" in q1:
        q2b = st.radio(
            "**Q2b - Is there a technical contradiction?**\n\n*For example, making it stronger makes it heavier; more accurate means more expensive*",
            ["Select an answer...",
             "Yes - there is a clear technical contradiction",
             "No - I just need to design something to meet requirements"],
            key="q2b"
        )
        if q2b != "Select an answer...":
            show_direct("TRIZ" if "Yes" in q2b else "DMADV")

    else:
        q2a = st.radio(
            "**Q2a - Has the problem already occurred?**\n\n*For example, customer complaint received, batch rejected, NCR issued*",
            ["Select an answer...",
             "Yes - a failure has already occurred",
             "No - the process is running but below potential"],
            key="q2a"
        )

        if q2a != "Select an answer...":

            if "Yes" in q2a:
                q3a = st.radio(
                    "**Q3a - Is there urgency of containment?**\n\n*For example, customer actively affected, production stopped, regulatory pressure*",
                    ["Select an answer...",
                     "Yes - urgent, customer or production at risk",
                     "No - problem occurred but no immediate urgency"],
                    key="q3a"
                )
                if q3a != "Select an answer...":
                    if "Yes" in q3a:
                        show_direct("8D")
                    else:
                        st.info("Candidates for TOPSIS ranking: **DMAIC** and **A3**")
                        show_topsis_stage(["DMAIC", "A3"])

            else:
                q3b = st.radio(
                    "**Q3b - What is the primary focus?**",
                    ["Select an answer...",
                     "Flow and waste - eliminate non-value-added activities, improve process flow",
                     "Quality and variability - reduce defects, variation, improve process capability"],
                    key="q3b"
                )
                if q3b != "Select an answer...":
                    if "Flow" in q3b:
                        candidates = ["Lean", "PDCA"]
                    else:
                        candidates = ["DMAIC", "A3"]
                    st.info(f"Candidates for TOPSIS ranking: **{candidates[0]}** and **{candidates[1]}**")
                    show_topsis_stage(candidates)
