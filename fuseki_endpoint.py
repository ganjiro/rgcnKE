
from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://localhost:3030/myFuseki/query")

# query = "select distinct ?p ?o from <http://www.disit.org/km4city/resource/Eventi/Eventi_a_Firenze> where {service <https://servicemap.disit.org/WebAppGrafo/sparql> {<http://www.disit.org/km4city/schema#Event> ?p ?o.}}"
query = """
        PREFIX km4c: <http://www.disit.org/km4city/schema#>    
        select distinct ?p ?o from <http://www.disit.org/km4city/resource/Eventi/Eventi_a_Firenze> where 
        {service <https://servicemap.disit.org/WebAppGrafo/sparql>
        {<http://www.disit.org/km4city/schema#Event> ?p ?o.}}
        """
sparql.setQuery(query)
sparql.setReturnFormat(JSON)
risQuery1 = sparql.query().convert()
print("we")
PROVA = 1
print("we2")