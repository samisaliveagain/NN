In this project, you are predicting the compressive strength of concrete — which is how strong a concrete mix is (in MPa) after it cures.


FROM THE DATA :

INPUT:
Cement				Cement content			g/m³
Blast Furnace Slag		Blast furnace slag		kg/m³
Fly Ash				Fly ash content			kg/m³
Water				Water content			kg/m³
Superplasticizer		Superplasticizer content	kg/m³
Coarse Aggregate		Coarse aggregate		kg/m³
Fine Aggregate			Fine aggregate			kg/m³
Age				Age of the concrete		Days

OUTPUT:
Concrete 			compressive strength		MPa



We take the data and pre process it 
1. We load the data 
2. Split features and target
3. Normalization
4. Train - test Split


Fully Connected Neural Network with Dropout (used for Monte Carlo sampling) has been used 
