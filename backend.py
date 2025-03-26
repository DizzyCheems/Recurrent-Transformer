import ollama
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def refine_response(input_seq, attentionstream_response):
    """
    Refines the AttentionStream response using Llama3 via ollama.
    Returns a refined response limited to 20 words, without meta-commentary.
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        refinement_prompt = (
            f"Given the question '{input_seq}', refine this response: '{attentionstream_response}'. "
            f"Provide only the refined response, without any meta-commentary like 'Here is...' or explanations."
        )
        response = await loop.run_in_executor(pool, lambda: ollama.chat(model="llama3", messages=[{"role": "user", "content": refinement_prompt}]))
        refined_text = (response['message']['content'].replace('\n', ' ').strip() 
                        if 'message' in response and 'content' in response['message'] 
                        else "Sorry, I couldn't refine the response.")
        words = refined_text.split()
        return ' '.join(words[:20])