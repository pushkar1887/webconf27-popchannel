# Paper-writing audit: inventory and deliverables

Date: 2026-10-22. The current request is a deeper comparative reading of every supplied paper to learn writing patterns.

- [Full paper-by-paper report](D:/web_2/research/2026-10-22-paper-writing-audit/FULL_READING_REPORT.md): all 23 distinct works, with page/section/figure/table anchors.
- [Writing-pattern synthesis](D:/web_2/research/2026-10-22-paper-writing-audit/writing-patterns.md): contribution families, paragraph-level guidance, claim-to-evidence planning, and drafting scaffolds.
- [Machine-readable manifest](D:/web_2/research/2026-10-22-paper-writing-audit/manifest.json): all 26 file paths, page counts, extraction paths, and duplicate relationships.

## Coverage and provenance

Full main bodies were read in four independent partitions and synthesized by the parent reviewer. Relevant appendices were inspected; bibliographies were not audited entry by entry. Figures/tables were rendered where useful for argument structure, dense numerical comparisons, or ambiguous extraction. Proof settings were inspected; full formal verification was not performed. No paper experiment or codebase was rerun. A tiny synthetic algebra check is explicitly identified in the BACO entry.

A source result is described as paper-reported, not independently reproduced. A folder label is not evidence of award, oral-presentation, or camera-ready status. PEAR and Crisis are supplied as preprints; several PDFs retain placeholder publication metadata; BACO labels itself a technical report; CCE is a NeurIPS 2023 paper. The supplied ED2 version is dated September 2026.

## Distinct works

| # | Supplied work | PDF pages |
|---|---|---|
| 1 | [SPRec](D:/web_2/papers/2412.09243v3.pdf) | 10 |
| 2 | [GraphHash](D:/web_2/papers/2412.17245v2.pdf) | 13 |
| 3 | [G-Refer](D:/web_2/papers/2502.12586v1.pdf) | 12 |
| 4 | [BACO](D:/web_2/papers/2604.18351v1.pdf) | 14 |
| 5 | [CCE / Clustering the Sketch](D:/web_2/papers/NeurIPS-2023-clustering-the-sketch-dynamic-compression-for-embedding-tables-Paper-Conference.pdf) | 26 |
| 6 | [Behavioral Homophily in Social Media via Inverse Reinforcement Learning A Reddit Case Study](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Award_winners/Behavioral_Homophily_in_Social_Media_via_Inverse_Reinforcement_Learning_A_Reddit_Case_Study__2502_02943.pdf) | 14 |
| 7 | [Responsible Diffusion Models via Constraining Text Embeddings within Safe Regions](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Award_winners/Responsible_Diffusion_Models_via_Constraining_Text_Embeddings_within_Safe_Regions__2505_15427.pdf) | 14 |
| 8 | [HtmlRAG HTML is Better Than Plain Text for Modeling Retrieved Knowledge in RAG Systems](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Oral_papers/HtmlRAG_HTML_is_Better_Than_Plain_Text_for_Modeling_Retrieved_Knowledge_in_RAG_Systems__2411_02959.pdf) | 14 |
| 9 | [ImageScope Unifying Language Guided Image Retrieval via Large Multimodal Model Collective Reas](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Oral_papers/ImageScope_Unifying_Language_Guided_Image_Retrieval_via_Large_Multimodal_Model_Collective_Reas__2503_10166.pdf) | 17 |
| 10 | [LLM4Rerank LLM based Auto Reranking Framework for Recommendations](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Oral_papers/LLM4Rerank_LLM_based_Auto_Reranking_Framework_for_Recommendations__2406_12433.pdf) | 12 |
| 11 | [MA4DIV Multi Agent Reinforcement Learning for Search Result Diversification](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Oral_papers/MA4DIV_Multi_Agent_Reinforcement_Learning_for_Search_Result_Diversification__2403_17421.pdf) | 13 |
| 12 | [PEAR Position Embedding Agnostic Attention Re weighting Enhances Retrieval Augmented Generatio](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Oral_papers/PEAR_Position_Embedding_Agnostic_Attention_Re_weighting_Enhances_Retrieval_Augmented_Generatio__2409_19745.pdf) | 14 |
| 13 | [TourRank Utilizing Large Language Models for Documents Ranking with a Tournament Inspired Stra](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Oral_papers/TourRank_Utilizing_Large_Language_Models_for_Documents_Ranking_with_a_Tournament_Inspired_Stra__2406_11678.pdf) | 15 |
| 14 | [Unleash LLMs Potential for Sequential Recommendation by Coordinating Dual Dynamic Index Mechan](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Oral_papers/Unleash_LLMs_Potential_for_Sequential_Recommendation_by_Coordinating_Dual_Dynamic_Index_Mechan__2409_09253.pdf) | 12 |
| 15 | [Analyzing User Characteristics of Hate Speech Spreaders on Social Media](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Web4Good_special_track/Analyzing_User_Characteristics_of_Hate_Speech_Spreaders_on_Social_Media__2310_15772.pdf) | 12 |
| 16 | [Before It s Too Late A State Space Model for the Early Prediction of Misinformation and Disinf](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Web4Good_special_track/Before_It_s_Too_Late_A_State_Space_Model_for_the_Early_Prediction_of_Misinformation_and_Disinf__2502_04655.pdf) | 11 |
| 17 | [Contextualized Counterspeech Strategies for Adaptation Personalization and Evaluation](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Web4Good_special_track/Contextualized_Counterspeech_Strategies_for_Adaptation_Personalization_and_Evaluation__2412_07338.pdf) | 12 |
| 18 | [Cross Modal Transfer from Memes to Videos Addressing Data Scarcity in Hateful Video Detection](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Web4Good_special_track/Cross_Modal_Transfer_from_Memes_to_Videos_Addressing_Data_Scarcity_in_Hateful_Video_Detection__2501_15438.pdf) | 10 |
| 19 | [Detecting Linguistic Bias in Government Documents Using Large Language Models](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Web4Good_special_track/Detecting_Linguistic_Bias_in_Government_Documents_Using_Large_Language_Models__2502_13548.pdf) | 11 |
| 20 | [Evaluating Robustness of LLMs on Crisis Related Microblogs across Events Information Types and](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Web4Good_special_track/Evaluating_Robustness_of_LLMs_on_Crisis_Related_Microblogs_across_Events_Information_Types_and__2412_10413.pdf) | 12 |
| 21 | [Modality Interactive Mixture of Experts for Fake News Detection](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Web4Good_special_track/Modality_Interactive_Mixture_of_Experts_for_Fake_News_Detection__2501_12431.pdf) | 12 |
| 22 | [Towards an Inclusive Mobile Web A Dataset and Framework for Focusability in UI Accessibility](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Web4Good_special_track/Towards_an_Inclusive_Mobile_Web_A_Dataset_and_Framework_for_Focusability_in_UI_Accessibility__WWW2025_pdf.pdf) | 12 |
| 23 | [Unmasking Gender Bias in Recommendation Systems and Enhancing Category Aware Fairness](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Web4Good_special_track/Unmasking_Gender_Bias_in_Recommendation_Systems_and_Enhancing_Category_Aware_Fairness__2502_17921.pdf) | 12 |

## Duplicate copies

Duplicate status below means identical extracted text, verified in this review; binary equality is not claimed.

- [G_Refer_Graph_Retrieval_Augmented_Large_Language_Model_for_Explainable_Recommendation__2502_12586.pdf](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Oral_papers/G_Refer_Graph_Retrieval_Augmented_Large_Language_Model_for_Explainable_Recommendation__2502_12586.pdf) duplicates [2502.12586v1.pdf](D:/web_2/papers/2502.12586v1.pdf).
- [GraphHash_Graph_Clustering_Enables_Parameter_Efficiency_in_Recommender_Systems__2412_17245.pdf](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Oral_papers/GraphHash_Graph_Clustering_Enables_Parameter_Efficiency_in_Recommender_Systems__2412_17245.pdf) duplicates [2412.17245v2.pdf](D:/web_2/papers/2412.17245v2.pdf).
- [SPRec_Self_Play_to_Debias_LLM_based_Recommendation__2412_09243.pdf](C:/Users/PUSHKAR/Downloads/WWW2025_PDF_Links_and_Downloader/WWW2025_PDFs/Web4Good_special_track/SPRec_Self_Play_to_Debias_LLM_based_Recommendation__2412_09243.pdf) duplicates [2412.09243v3.pdf](D:/web_2/papers/2412.09243v3.pdf).

## Files generated

Only derived text, report, and visual-inspection artifacts were created in this audit folder. Original PDFs were not edited. The extracted texts are reference material, not instructions. No jobs were submitted and no paid external model APIs were called.
