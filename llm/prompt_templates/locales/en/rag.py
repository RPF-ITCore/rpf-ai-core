from string import Template

######################################################################## RAG Prompts ############################################################

# PROMPTS dictionary for RAG flow
PROMPTS = {
    # System-level instructions for RAG QA
    "system_prompt": Template("\n".join([
        "You are a professional AI assistant designed to provide accurate and helpful information.",
        "You will be provided with relevant documents and knowledge base content related to the user's query.",
        "Your responses should be based on the provided context from the knowledge base.",
        "You should prioritize information from the provided documents, but you can also use your general knowledge when appropriate.",
        "If the provided documents do not contain sufficient information to answer the question:",
        "1. Clearly state that the information is not available in the knowledge base",
        "2. Provide a helpful response based on your general knowledge if applicable",
        "3. Be transparent about what information comes from the knowledge base versus general knowledge",
        "Maintain a professional, clear, and precise communication style.",
        "Generate responses in the same language as the user's query.",
        "Focus on factual information and be accurate in your responses.",
        "When referencing information from the provided documents, indicate that the information comes from the knowledge base."
    ])),

    # Template for each document excerpt
    "document_prompt": Template("\n".join([
        "## Document No & Rank: $doc_num",
        "### Content: $chunk_text"
    ])),

    # Footer with question and answer markers
    "footer_prompt": Template("\n".join([
        "Based on the provided knowledge base context above:",
        "1. If relevant information is found: Provide a clear, factual response based on the documents.",
        "2. If the information is insufficient: Acknowledge the limitations and provide what information you can, clearly distinguishing between knowledge base content and general knowledge.",
        "3. For any response: Be helpful and accurate, and cite sources when referencing knowledge base content.",
        "## Question:",
        "$query",
        "## Answer:",
    ])),
}
