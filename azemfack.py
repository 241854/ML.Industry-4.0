import pandas as pd 
import streamlit as st
#import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from fontawesome import icons
from pycaret.regression import setup as setup_reg
from pycaret.regression import compare_models as compare_models_reg
from pycaret.regression import save_model as save_model_reg
from pycaret.regression import plot_model as plot_model_reg
from pycaret.classification import setup as setup_class
from pycaret.classification import compare_models as compare_models_class
from pycaret.classification import save_model as save_model_class
from pycaret.classification import plot_model as plot_model_class
from pycaret.clustering import setup as setup_clust
from pycaret.clustering import create_model as create_model_clust
from pycaret.clustering import assign_model as assign_model_clust
from pycaret.anomaly import setup as setup_anomaly
from pycaret.anomaly import create_model as create_model_anomaly
from pycaret.anomaly import assign_model as assign_model_anomaly
from pycaret.nlp import setup as setup_nlp
from pycaret.nlp import create_model as create_model_nlp
from pycaret.nlp import assign_model as assign_model_nlp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

url= "https://github.com/241854/DATA-SCIENTIST_CON_R.git"

@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

# Definisci le credenziali valide
valid_credentials = {
    "pages": "azaipa",
    "username2": "password2",
}

# Funzione per ottenere lo stato di sessione
def get_session_state():
    return st.session_state

def main():
    session_state = get_session_state()  # Ottieni lo stato di sessione

    st.title("MACHINE LEARNING INDUSTRY(4.0)")
    st.image("https://hooshio.com/wp-content/uploads/2022/08/b2.jpg")
    # Cambio del colore della sidebar
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #00FF00; /* Cambia il colore qui */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.image("https://sis.id.ethz.ch/services/datascience/images/machine_learning_icon_notext.png")
    st.sidebar.header("scopriamo il nascoto, rispaemiamo in tempo e soldi")
    st.sidebar.write("[Autore: PAGES AIME AZEMFACK](%s)"% url)
    
    # Imposta lo stato di accesso iniziale
    if "logged_in" not in session_state:
        session_state.logged_in = False

    # Interfaccia di login solo se l'utente non è già loggato
    if not session_state.logged_in:
        access_option = st.sidebar.radio("Accesso", ["login"])
        
        if access_option == "login":
            col1, col2 = st.sidebar.columns(2)
            with col1:
                username = st.text_input("Nome Utente")
            with col2:
                password = st.text_input("Password", type="password")

            # Verifica le credenziali solo se l'utente sceglie di accedere
            if st.sidebar.button("Accedi"):
                if username in valid_credentials and valid_credentials[username] == password:
                    st.sidebar.success("Accesso riuscito!")
                    session_state.logged_in = True
                else:
                    st.sidebar.error("Nome utente o password non validi. Riprova.")

    # Se l'utente ha effettuato l'accesso o la registrazione, mostra il menu
    if session_state.logged_in:
        show_menu()

def show_menu():
    st.sidebar.header("MENU")
     
    # Definizione del menu principale
    main_menu = ["Home", "Visualizzazione Dati", "Analisi Statistica", "Machine Learning", "Informazioni"]

    # Selezione dell'opzione del menu
    selected_menu_option = st.sidebar.radio("scegli l'operazione da effettuare", main_menu)

    # Contenuto delle diverse sezioni
    if selected_menu_option == "Home":
        show_home_page()

    elif selected_menu_option == "Visualizzazione Dati":
        show_data_visualization_page()

    elif selected_menu_option == "Analisi Statistica":
        show_statistical_analysis_page()

    elif selected_menu_option == "Machine Learning":
        show_machine_learning_page()

    elif selected_menu_option == "Informazioni":
        show_information_page()
  
        
        
def show_home_page():
    
    task = st.selectbox("Seleziona un'opzione", ["OBBIETTIVO", "BENEFICI", "USO"]) 
                                                                        
    if task == "OBBIETTIVO":
        st.header("Quest'applicazione ti aiuterà a pianificare, identificare, predire, classificare, d'ottimizzare il tempo per la resoluzione delle anormalie dell'impianto")
        st.write("Inanzitutto, rilevamento delle anomalie in tempo reale dei pezzi attraverso la visualizzazione."
                " Poi, la classificazione dei tipi di anormalia (alarmi)."
                " la predizione delle future anomalie e il ciclo di vita del pezzo"
             " apprendimento automatico con scopo di risolvere il problema da solo o orientare il manutentore dove c'è il guasto.")
        st.sidebar.markdown("- **Cellulare**: +39 324 891 7568")
        st.sidebar.markdown("- **Email**: [azaipasrl@gmail.com](mailto:azaipasrl@gmail.com)")
        st.sidebar.markdown("- **Link per approfondire le tue ricerche in ML industry 4.0**: Fai clic qui per accedere: https://www.valispace.com/how-machine-learning-is-helping-engineers-with-predictive-maintenance-and-prevent-equipment-failures/")    
    elif task == "BENEFICI":
        st.header("Vantaggi di ML in automation Engineering")
        st.write("Preallarme e manutenzione proattiva."
                "Risparmio sui costi e maggiore efficienza: Machine Learning può risparmiare tempo e denaro riducendo i tempi di inattività e aumentando la produttività."
                "Disponibilità e qualità dei dati: è essenziale raccogliere dati per gli algoritmi di apprendimento automatico. Queste informazioni devono essere facilmente accessibili in numeri sufficienti per gli algoritmi per imparare correttamente.")

    elif task == "USO":
        st.header("come utillizzare quest'applicazione")
        st.write("1-Inserisci il tuo credenziali. 2-scegli un opzione del menu secondo cioìò che vuoi analizzare. 3- scaricare il tuo file csv. 4-clicca su analizzare i dati o una scelta per quanto riguarda ML")

def show_data_visualization_page():
    st.header("Visualizza i tuoi Dati rapidamente e simplicemente con noi!")
    st.write("Qui puoi esplorare e visualizzare i dati utilizzati nell'analisi.")
    file = st.file_uploader(label = "Carica il tuo file CSV", type = ["csv"])
    st.sidebar.markdown("**Nota: I tuoi dati non devono contenere valori mancanti (puoi pulirli con SQL o Python).**")  
    if file is not None:
        data = pd.read_csv(file)
        st.dataframe(data.head(10))
        profile = st.button("Visualizza i tuoi dati")
        if profile:
            profile_df = pandas_profiling.ProfileReport(data)
            st_profile_report(profile_df)
            if file is not None:
                st.download_button(label="Scarica file CSV", data=file, file_name='data.csv', mime='text/csv')

def show_statistical_analysis_page():
    st.header("Fai statistical analist con noi!")
    st.write("Effettua analisi statistiche sui dati per ottenere insight.")
    file = st.file_uploader(label = "Carica il tuo file CSV", type = ["csv"])
    if file is not None:
        data = pd.read_csv(file)
        st.dataframe(data.head(10))
        profile = st.button("Visualizza i tuoi dati")

def show_machine_learning_page():
    st.header("Ecco il tuo spazio d apprendimento automatico(Machine Learning)")
    st.write("Utilizza algoritmi di Machine Learning per addestrare modelli e fare le tue predizioni.")
    
    # Opzioni di Machine Learning
    file = st.file_uploader(label = "Carica il tuo file CSV", type = ["csv"])
    st.sidebar.markdown("**Nota: I tuoi dati non devono contenere valori mancanti (puoi pulirli con SQL o Python).**")  
    if file is not None:
        data = pd.read_csv(file)
        st.dataframe(data.head(10))
        profile = st.button("Visualizza i tuoi dati")
        target = st.selectbox("Seleziona il target", data.columns)
        task = st.selectbox("Seleziona il tipo di Machine Learning / Deep Learning", ["Regresione", "Classificazione", 
                                                                        "Clustering", "Rilevamento Anomalie",
                                                                        "Elaborazione del Linguaggio Naturale (NLP)", "Neural Network (Deep Learning)"])
        if task == "Regresione":
            if st.button("Calcola il modello"):
                exo_reg = setup_reg(data, target = target)
                model_reg = compare_models_reg()
                save_model_reg(model_reg, "best_model_regression.pkl")
                st.success("Modello di Regressione generato con successo!!!")
                
                st.write("Risidui")
                plot_model_reg(model_reg, plot = "residui", save = True)
                st.image("risidui.png")
                
                st.write("Feature Importance")
                plot_model_reg(model_reg, plot = "feature", save = True)
                st.image("feature_importance.png")
                
                with open("best_reg_model.pkl", "rb") as f:
                    st.download_button("Scarica il modello (pipline)", f, file_name = "best_reg_model.pkl")
 
        elif task == "Classificazione":
            if st.button("Calcola il modello"):
                exp_reg = setup_class(data, target = target)
                model_class = compare_models_class()
                save_model_class(model_class, "best_model_class.pkl")
                st.success("Modello di Classificazione generato con successo!!!")
                
                st.write("ROC curve")
                plot_model_class(model_class, save = True)
                st.image("ROC_curve.png")
                
                st.write("Classification Report")
                plot_model_class(model_class, plot = "class_report", save = True)
                st.image("Classification_Report.png")
                
                st.write("Confusion Matrix")
                plot_model_class(model_class, plot = "confusion_matrix", save = True)
                st.image("Confusion_Matrix.png")
                    
                st.write("Feature Importance")
                plot_model_class(model_class, plot = "feature", save = True)
                st.image("Feature_Importance.png")
                
                with open("best_class_model.pkl", "rb") as f:
                    st.download_button("Scarica il modello Classificazione", f, file_name = "best_class_model.pkl")
                
        elif task == "Clustering":
            if st.button("Esegui Clustering"):
                # Esegui il setup
                clust_setup = setup_clust(data)

                # Crea il modello
                clustering_model = create_model_clust()

                # Assegna il modello
                clustered_data = assign_model_clust(clustering_model)

                # Visualizza i risultati
                st.write("Dati clusterizzati:", clustered_data)  
            
        elif task == "Rilevamento Anomalie":
            if st.button("Esegui il rilevamento delle anomalie"):
                # Esegui il setup
                anomaly_setup = setup_anomaly(data)

                # Crea il modello
                anomaly_model = create_model_anomaly()

                # Assegna il modello
                anomalous_data = assign_model_anomaly(anomaly_model)

                # Visualizza i risultati
                st.write("Dati anomali:", anomalous_data)
            
        elif task == "Elaborazione del Linguaggio Naturale (NLP)":
            if st.button("Esegui il NLP"):
                # Esegui il setup
                nlp_setup = setup_nlp(data)

                # Crea il modello
                nlp_model = create_model_nlp()

                # Assegna il modello
                processed_data = assign_model_nlp(nlp_model)

                # Visualizza i risultati
                st.write("Dati elaborati:", processed_data) 
         
        elif task == "Neural Network (Deep Learning)":
            if st.button("Esegui il Deep Learning"):
                # Codice per il deep learning con TensorFlow
                st.write("In questa sezione, puoi addestrare modelli di deep learning utilizzando reti neurali.")

                # Esempio di un modello di rete neurale convoluzionale (CNN) con TensorFlow
                model = Sequential([
                    Conv2D(16, 3, padding='same', activation='relu', input_shape=(28, 28, 1)),
                    Flatten(),
                    Dense(10, activation='softmax')
                ])

                # Visualizza la struttura del modello
                st.write("Struttura del modello:")
                model.summary()

                # Compila il modello
                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                # Il resto del codice per il Machine Learning  

def show_information_page():
    st.header("Informazioni sull'Applicazione")
    st.write("Qui puoi trovare ulteriori informazioni sull'applicazione e come utilizzarla.")
    st.subheader("Contatti e Ulteriori Informazioni")
    st.write("Grazie per utilizzare la nostra applicazione 'MACHINE LEARNING INDUSTRY(4.0)'. Per qualsiasi domanda, feedback o richiesta di supporto, non esitare a contattarci.")
    
    st.markdown("- **Cellulare**: +39 324 891 7568")
    st.markdown("- **Email**: [azaipasrl@gmail.com](mailto:azaipasrl@gmail.com)")
    st.markdown("- **Link per ulteriori risorse**: Fai clic qui per ulteriori informazioni: https://www.valispace.com/how-machine-learning-is-helping-engineers-with-predictive-maintenance-and-prevent-equipment-failures/")


# Il resto del tuo codice rimane invariato
if __name__=="__main__":
    main()
