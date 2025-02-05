import gradio as gr
import joblib
# from preprocessing import clean_text

def load_models():    
    with open('./models/model.pkl', 'rb') as model_file:
        model = joblib.load(model_file)

    # Carregar o vectorizer
    with open('./models/vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = joblib.load(vectorizer_file)
    return model,vectorizer

def classify_text(text):
    if not text:
        return "Forneça um texto para classificar"
    try: 
        cleaned_text = text #clean_text(text)

        model, vectorizer = load_models()
        X = vectorizer.transform([cleaned_text])

        if hasattr(X, "toarray"):  # Verifica se o método 'toarray' existe
            X = X.toarray()

        prediction = model.predict(X)
        
        # Ajustar para formato do output do modelo
        result = f"Classificação: {'Escrita por LLM' if prediction == 1 else 'Escrita por Humano'}\n"

        return result
    except Exception as e:
        return f"Erro ao processar o texto: {str(e)}"
    
interface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(label="Insira a redação", placeholder="Digite o texto aqui..."),
    outputs=gr.Textbox(label="Resultado da Classificação"),
    title="Classificador de Redações",
    description="Este modelo identifica se uma redação foi escrita por um humano ou por uma LLM.",
)

if __name__ == "__main__":
    interface.launch(share=True, debug=True)
