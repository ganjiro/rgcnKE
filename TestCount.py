from SPARQLWrapper import SPARQLWrapper, JSON, TSV


sparql = SPARQLWrapper("http://localhost:3030/myFuseki/query")
sparql.setQuery("""PREFIX km4c: <http://www.disit.org/km4city/schema#>
                  select count(*) from  <http://www.disit.org/km4city/resource/Eventi/Eventi_a_Firenze> 
                  where {service <https://servicemap.disit.org/WebAppGrafo/sparql>  {?s ?p ?o}}

                  """)
sparql.setReturnFormat(JSON)
risQuery1 = sparql.query().convert()


