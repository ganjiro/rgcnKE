
from SPARQLWrapper import SPARQLWrapper, JSON, N3, XML
from rdflib import Graph

'''
# seleziona tutte le singole entità di tipo Event:
select distinct ?s from  <http://www.disit.org/km4city/resource/Eventi/Eventi_a_Firenze> where 
{service <https://servicemap.disit.org/WebAppGrafo/sparql>  {?s a km4c:Event}
}
'''

sparql = SPARQLWrapper("https://servicemap.disit.org/WebAppGrafo/sparql")
sparql.setQuery("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX km4c: <http://www.disit.org/km4city/schema#>
    prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>    
    SELECT DISTINCT ?o FROM  <http://www.disit.org/km4city/resource/Eventi/Eventi_a_Firenze> WHERE {SERVICE <https://servicemap.disit.org/WebAppGrafo/sparql>  {?s ?p ?o. ?s a km4c:Event FILTER (?p = <http://www.disit.org/km4city/schema#eventCategory>)}}
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
for result in results["results"]["bindings"]:
      print(result["o"]["value"])
# sparql.setReturnFormat(N3)
# results = sparql.query().convert()
# g = Graph()
# g.parse(data=results, format="n3")
# print(g.serialize(format='n3'))
# sparql.setReturnFormat(XML)
# results = sparql.query().convert()
# print(results.serialize(format='xml'))
a = 4