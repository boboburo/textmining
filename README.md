Supporting Project for Text Mining Presentation 
===============================================

## Presentation notes

Text mining: Finding hidden information in CRM data

Description: A client in the energy sector wanted to create predictive
behavioural models of business customers at the company level, but the
CRM data was messy, often containing several sub-accounts for each business,
without any grouping identifiers, and so aggregation was impossible. 

In this talk I describe a short project where we used text mining, 
a handful of unsupervised learning techniques, and pragmatic use of 
human skill under time & budgetary constraints, to identify the true 
company level structures in the CRM data. 


## Data notes

Obviously the client data is confidential, so I've created a dummy dataset 
based on a list of enforced NAMA properties which has similar features
to the client dataset.

Available as a Google Doc here: 
https://docs.google.com/spreadsheet/ccc?key=0AjOXYk-Wh9M2dGt1TWxjZzc5dDBtY29NMjRvUDhUTXc


## Dev environment notes

If using Anaconda distro, create a new virtualenv using the below 

	conda create -n textmining --file requirements_conda.txt


To use/exit the Anaconda environment:
	
	source activate textmining
	source deactivate

Run the ipython notebook from the `textmining/` directory:    

	ipython notebook

