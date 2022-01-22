# SI_EXPLAINER_tg_bot  
This bot is an assistant for medical professionals in interpreting the results of patient clustering.  
  
# ABOUT  
This chatbot was developed specifically for medical experts as an assistant in the interpretation of patient clustering results. The chatbot is based on the algorithm described in the article [3]. This method is based on statistical inference and allows to get the description of the clusters, determining the influence of a particular feature on the difference between them. Based on the proposed approach, it is possible to determine the characteristic features for each cluster. More details on ideas and approaches to interpretation can be found in the articles [1,2].  
  
# MAIN COMMANDS  
**/start** – receive a greeting from the bot and its brief description.  
**/help** – receive a file with an instruction manual.  
**/why** – the bot explains why the patient, which corresponds to the specified id, got or did not get into the specified cluster.  
**/getfile** – if a file was loaded for interpretation without marking by clusters, then the bot will mark it up on its own and, after issuing this command, will return the file to the user with already marked data.  
  
# REFERENCES  
[1] Balabaeva K, Kovalchuk S. Post-hoc Interpretation of clinical pathways clustering using Bayesian inference. Procedia Computer Science. 178 (2020), 264-273.  
[2] Balabaeva K, Kovalchuk S. Clustering Results Interpretation of Continuous Variables Using Bayesian Inference. Studies in Health Technology and Informatics. 281 (2021), 477-481  
[3] Kanonirov A., Balabaeva K.Y., Kovalchuk S. Statistical inference for clustering results interpretation in clinical practice//Studies in health technology and informatics, 2021, Vol. 285, pp. 100-105  
  
![image](https://user-images.githubusercontent.com/63186837/150526570-3ef57fdd-0d5d-48f0-b170-9bd6958c2809.png)


