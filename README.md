                                                              INTRODUZIONE 


Questo lavoro è collocato nel campo della scienza dei dati con applicazione al settore della manutenzione predittiva. La necessità di avere un modo per determinare se una particolare macchina fallirà o meno, così come la natura del guasto, è essenziale per la generazione  industrie 4.0. La ragione principale risiede nella seguente considerazione: la riparazione o la sostituzione di una macchina difettosa richiede generalmente costi molto più elevati di quelli richiesti per la sostituzione di un singolo componente. Pertanto, l'installazione di sensori che monitorano lo stato delle macchine, raccogliendo le informazioni appropriate, può portare a grandi risparmi per le industrie.

Qui utilizziamo il set di dati di manutenzione predittiva AI4I dal repository UCI per effettuare un'analisi che mira a rispondere alle esigenze appena segnalate. In particolare, il lavoro viene presentato attraverso una lineup che caratterizza una tipica applicazione di Machine Learning. In primo luogo il set di dati viene esplorato per ottenere una conoscenza più profonda che può guidare nella piena comprensione della verità di base. Poi, alcune tecniche di pre-elaborazione vengono applicate per preparare i dati per gli algoritmi che useremo per fare le nostre previsioni. Consideriamo due compiti principali: il primo consiste nello stabilire se una macchina generica sta per subire un guasto mentre il secondo riguarda la determinazione della natura del guasto. Infine, viene fornito un confronto tra i risultati ottenuti da questi ultimi, valutando sia le loro prestazioni attraverso metriche appropriate, sia la loro interpretabilità.

                                                                tabella dei contenuti
1.	Task and Data Description
2.	Exploratory Analysis
2.1 ID Columns
2.2 Target anomalies
2.3 Outliers Inspection
2.4 Resampling with SMOTE
2.5 Comparison after resampling
2.6 Features scaling and Encoding
2.7 PCA and Correlation Heatmap
2.8 Metrics
3.	Binary task
3.1 Preliminaries
3.2 Feature Selection attempts
3.3 Logistic Regression Benchmark
3.4 Models
4.	Multi-class task
4.1 Logistic Regression Benchmark
4.2 Models
5.	Decision Paths
6.	Conclusions
