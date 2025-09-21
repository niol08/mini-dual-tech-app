# # utils/page.py
# import streamlit as st
# from utils.llm import upload_to_gemini, chat_once

# NON_GEMINI_TYPES = {"edf", "bdf", "fif", "set"}  # don't upload raw EEG to Gemini

# def biosignal_chat_page(
#     biosignal_label: str,
#     slug: str,
#     accepted_types=("pdf", "txt", "csv", "edf", "fif", "set"),
#     analyzer=None,                    # NEW
#     analyzer_label="Analyze",         # NEW
# ):
#     st.title(f"{biosignal_label} Chat")

#     with st.sidebar:
#         st.header(biosignal_label)
#         st.caption("Upload a relevant document (report or data file), then ask questions.")
#         if st.button("🔄 Reset chat", key=f"reset_{slug}"):
#             for key in [f"{slug}_messages", f"{slug}_file_obj", f"{slug}_file_name", f"{slug}_analysis"]:
#                 if key in st.session_state:
#                     del st.session_state[key]
#             st.rerun()

#     uploaded = st.file_uploader(
#         f"Upload a {biosignal_label} document",
#         type=list(accepted_types),
#         key=f"uploader_{slug}",
#         accept_multiple_files=False,
#     )

#     if uploaded is not None and st.session_state.get(f"{slug}_file_name") != uploaded.name:
#         ext = (uploaded.name.split(".")[-1] or "").lower()
#         if ext not in NON_GEMINI_TYPES:
#             with st.status("Uploading file to Gemini…"):
#                 file_obj = upload_to_gemini(uploaded)
#             st.session_state[f"{slug}_file_obj"] = file_obj
#         else:
#             st.session_state[f"{slug}_file_obj"] = None
#         st.session_state[f"{slug}_file_name"] = uploaded.name
#         st.success(f"Attached: {uploaded.name}")

#     if st.session_state.get(f"{slug}_file_name"):
#         st.caption(f"📎 Attached: **{st.session_state[f'{slug}_file_name']}**")

#     # ---- NEW: run analyzer on demand ----
#     if analyzer and uploaded is not None:
#         if st.button(analyzer_label, key=f"analyze_{slug}"):
#             with st.spinner("Running analysis…"):
#                 result = analyzer(uploaded)
#             st.session_state[f"{slug}_analysis"] = result

#     if st.session_state.get(f"{slug}_analysis"):
#         st.subheader("Analysis result")
#         st.json(st.session_state[f"{slug}_analysis"])
        
#                 # ---- Optional: visualize seizure probabilities over time if present ----
#         res = st.session_state[f"{slug}_analysis"]
#         try:
#             if isinstance(res, dict) and "per_window" in res and res["per_window"]:
#                 p_seiz = [w.get("p_seiz") for w in res["per_window"] if "p_seiz" in w]
#                 times = [w.get("start_s", i) for i, w in enumerate(res["per_window"])]
#                 # headline metric
#                 pred = res.get("prediction", "?")
#                 conf = res.get("confidence", None)
#                 if conf is not None:
#                     st.metric("Model prediction", pred, delta=f"confidence {conf:.3f}")
#                 # quick plot
#                 import matplotlib.pyplot as plt
#                 fig, ax = plt.subplots(figsize=(8, 2.5))
#                 ax.plot(times, p_seiz)
#                 ax.set_xlabel("Time (s)")
#                 ax.set_ylabel("p_seiz")
#                 ax.set_ylim(0, 1)
#                 ax.grid(True, alpha=0.3)
#                 st.pyplot(fig)
#         except Exception as _e:
#             st.caption("Plotting skipped (unexpected analysis format).")

        
#         # seed chat context with a compact summary once
#         if not st.session_state.get(f"{slug}_analysis_seeded"):
#             summary = f"EEG analysis summary:\n```\n{st.session_state[f'{slug}_analysis']}\n```"
#             msgs = st.session_state.setdefault(f"{slug}_messages", [])
#             msgs.append({"role": "assistant", "content": summary})
#             st.session_state[f"{slug}_analysis_seeded"] = True

#     # ---- Chat UI (unchanged) ----
#     messages = st.session_state.setdefault(f"{slug}_messages", [])
#     for m in messages:
#         with st.chat_message(m["role"]):
#             st.markdown(m["content"])

#     placeholder = f"Ask about your {biosignal_label} document or data…"
#     if user_text := st.chat_input(placeholder):
#         messages.append({"role": "user", "content": user_text})
#         with st.chat_message("user"):
#             st.markdown(user_text)
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking…"):
#                 reply = chat_once(
#                     biosignal_label=biosignal_label,
#                     history=messages[:-1],
#                     user_prompt=user_text,
#                     file_obj=st.session_state.get(f"{slug}_file_obj"),
#                 )
#             st.markdown(reply or "_(no response)_")
#         messages.append({"role": "assistant", "content": reply})


import streamlit as st
from utils.llm import upload_to_gemini, chat_once

NON_GEMINI_TYPES = {"edf", "bdf", "fif", "set", "csv", "txt"}

def biosignal_chat_page(
    biosignal_label: str,
    slug: str,
    accepted_types=("pdf", "txt", "csv", "edf", "fif", "set"),
    analyzer=None,
    analyzer_label="Analyze",
):
    st.title(f"{biosignal_label} Chat")

    with st.sidebar:
        st.header(biosignal_label)
        st.caption("Upload a relevant document (report or data file), then ask questions.")
        if st.button("🔄 Reset chat", key=f"reset_{slug}"):
            for key in [f"{slug}_messages", f"{slug}_file_obj", f"{slug}_file_name", f"{slug}_analysis", f"{slug}_analysis_seeded"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    uploaded = st.file_uploader(
        f"Upload a {biosignal_label} document",
        type=list(accepted_types),
        key=f"uploader_{slug}",
        accept_multiple_files=False,
    )


    if uploaded is not None and st.session_state.get(f"{slug}_file_name") != uploaded.name:
        ext = (uploaded.name.split(".")[-1] or "").lower()
        if ext not in NON_GEMINI_TYPES:

            try:
                with st.status("Uploading file to Gemini…"):
                    file_obj = upload_to_gemini(uploaded)
                st.session_state[f"{slug}_file_obj"] = file_obj
            except Exception as e:
                st.session_state[f"{slug}_file_obj"] = None
                st.error(f"File upload to Gemini failed: {e}")
        else:
            st.session_state[f"{slug}_file_obj"] = None

        st.session_state[f"{slug}_file_name"] = uploaded.name
        st.success(f"Attached: {uploaded.name}")

    if st.session_state.get(f"{slug}_file_name"):
        st.caption(f"📎 Attached: **{st.session_state[f'{slug}_file_name']}**")

    if analyzer and uploaded is not None:
        if st.button(analyzer_label, key=f"analyze_{slug}"):
            with st.spinner("Running analysis…"):
                result = analyzer(uploaded)
            st.session_state[f"{slug}_analysis"] = result

            st.session_state[f"{slug}_analysis_seeded"] = False

    if st.session_state.get(f"{slug}_analysis"):
        st.subheader("Analysis result")
        st.json(st.session_state[f"{slug}_analysis"])


        res = st.session_state[f"{slug}_analysis"]
        try:
            if isinstance(res, dict) and "per_window" in res and res["per_window"]:
                p_seiz = [w.get("p_seiz") for w in res["per_window"] if "p_seiz" in w]
                times = [w.get("start_s", i) for i, w in enumerate(res["per_window"])]
                pred = res.get("prediction", "?")
                conf = res.get("confidence", None)
                if conf is not None:
                    st.metric("Model prediction", pred, delta=f"confidence {conf:.3f}")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 2.5))
                ax.plot(times, p_seiz)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("p_seiz")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        except Exception:
            st.caption("Plotting skipped (unexpected analysis format).")


        if not st.session_state.get(f"{slug}_analysis_seeded"):

            summary = f"{biosignal_label} analysis summary:\n```\n{st.session_state[f'{slug}_analysis']}\n```"
            msgs = st.session_state.setdefault(f"{slug}_messages", [])
            msgs.append({"role": "assistant", "content": summary})
            st.session_state[f"{slug}_analysis_seeded"] = True

    messages = st.session_state.setdefault(f"{slug}_messages", [])
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    placeholder = f"Ask about your {biosignal_label} document or data…"
    if user_text := st.chat_input(placeholder):
        messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):

                reply = chat_once(
                    biosignal_label=biosignal_label,
                    history=messages[:-1],
                    user_prompt=user_text,
                    file_obj=None,
                )
            st.markdown(reply or "_(no response)_")
        messages.append({"role": "assistant", "content": reply})
