from SPARQLWrapper import SPARQLWrapper, JSON, TSV

sparql = SPARQLWrapper("http://localhost:3030/myFuseki/query")

sparql.setQuery("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX km4c: <http://www.disit.org/km4city/schema#>
                select distinct ?o from  <http://www.disit.org/km4city/resource/Eventi/Eventi_a_Firenze> where {
                service <https://servicemap.disit.org/WebAppGrafo/sparql>  {?s ?p ?o  filter (?p != rdf:type)}} order by ?o
                """)
sparql.setReturnFormat(JSON)
risQuery1 = sparql.query().convert()

file = open("mycount_file.txt", "w")
#
# for risQuery1_itr in risQuery1["results"]["bindings"]:
#     sparql.setQuery("""PREFIX km4c: <http://www.disit.org/km4city/schema#>
#             select distinct ?s ?p ?o from
#                   <http://www.disit.org/km4city/resource/Eventi/Eventi_a_Firenze> where
#                   {service <https://servicemap.disit.org/WebAppGrafo/sparql>
#
#                     {{?s  ?p ?o.
#                     FILTER(?s = <""" + str(risQuery1_itr["s"]["value"]) + """>)
#                       }
#
#                       union
#                       {?s ?p ?o.
#                        <""" + str(risQuery1_itr["s"]["value"]) + """>  ?y ?s.
#                       }
#
#                         union
#                       {?s ?p ?o.
#                        ?x ?y ?s.
#                        <""" + str(risQuery1_itr["s"]["value"]) + """>  ?l ?x.
#                       }
#                       }
#                     }
#       """)
#     sparql.setReturnFormat(TSV)
#     risQuery2 = sparql.query().convert()
#     file.write(risQuery2.decode("UTF-8")[9:-1].replace('\t', ' ').replace('\n', ' .\n'))
#
# file.close()