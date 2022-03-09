# Node Classification e Link Prediction su Knowledge Graph tramite Graph Neural Network

Un package python che incapsula 2 modelli GNN-based per fare  rispettivamente node classification e link prediction su un knoledge graph a partire semplicemente dal sottografo delle triple in formato tsv. Vengono forniti inoltre due modelli di confronto non GNN-based. Questo progetto è stato svolto come esame e project work in Knowledge Engineering, presso l'Università degli Studi di Firenze, sotto la supervisione del Professore Bellini Pierfrancesco.
Questo progetto è stato svolto in collaborazione con [Marco Mistretta]

## Prerequisiti
- un PC con fuseki jena installato

## Idea
Esiste un solo file main accessibile, in tale file si può decidere quali modelli avviare e su quale dataset.
Per selezionare i parametri in ingresso dei modelli modificare i rispetttivi file .ini

## Usare il proprio dataset
Se si desidera utilizzare una propria knowledge base basterà caricarla in formato tsv nella cartella apposita, e specificare semplicemente l'entità o la label sulla quale si desidera fare node classification. Le funzioni contenute nel package ManageDataset si occuperanno del resto

## Help 
Per maggiori informazioni conattatare uno dei due collaboratori

### Last release
- fixato bug nella riformattazione del dataset 

[Marco Mistretta]: <https://github.com/marcomistretta>
