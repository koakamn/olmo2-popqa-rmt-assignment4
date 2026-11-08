Assignment 3 code package (PopQA / OLMo-2 training-stage study)
Run order: see Appendix A of the PDF. data/popQA.tsv is downloaded from
https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/main/data/popQA.tsv (MIT).
To obtain real LLM results run on a GPU (Colab):  python run_llm.py --condition base|sft|dpo|instruct --prompt_set 1
then: python analyze.py base,sft,dpo,instruct
