
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
                  select distinct ?s from  <http://www.disit.org/km4city/resource/Eventi/Eventi_a_Firenze> 
                  where {service <https://servicemap.disit.org/WebAppGrafo/sparql>  {?s a km4c:Event }}
                  
                  """)
sparql.setReturnFormat(JSON)
risQuery1 = sparql.query().convert()
file = open("testfile.txt","w")


for risQuery1_itr in risQuery1["results"]["bindings"]:
      sparql.setQuery("""
                select distinct ?p ?o from  
            <http://www.disit.org/km4city/resource/Eventi/Eventi_a_Firenze> where 
            {service <https://servicemap.disit.org/WebAppGrafo/sparql>  
              {<""" + str(risQuery1_itr["s"]["value"]) + """>?p ?o.
                                                  
                }
              }
            
      """)
      sparql.setReturnFormat(JSON)
      risQuery2 = sparql.query().convert()

      fail = 0
      for risQuery2_itr in risQuery2["results"]["bindings"]:
            try:
                  file.write(str(risQuery1_itr["s"]["value"]) + ' ' + str(risQuery2_itr["p"]["value"]) + ' ' + str(risQuery2_itr["o"]["value"]) + '.\n')
            except:
                  fail += 1

            if risQuery2_itr["o"]["type"] != 'uri': continue
            sparql1 = SPARQLWrapper("https://servicemap.disit.org/WebAppGrafo/sparql")
            sparql1.setQuery("""
                      select distinct ?p ?o from  
                  <http://www.disit.org/km4city/resource/Eventi/Eventi_a_Firenze> where 
                  {service <https://servicemap.disit.org/WebAppGrafo/sparql>  
                    {<""" + str(risQuery2_itr["o"]["value"]) + """>?p ?o.
                      }
                    }
            """)
            sparql1.setReturnFormat(JSON)
            risQuery3 = sparql1.query().convert()

            for risQuery3_itr in risQuery3["results"]["bindings"]:

                  try:
                        file.write(str(risQuery2_itr["o"]["value"]) + ' ' + str(risQuery3_itr["p"]["value"]) + ' ' + str(
                              risQuery3_itr["o"]["value"]) + '.\n')
                  except:
                        fail += 1
                  if risQuery2_itr["o"]["type"] != 'uri': continue
      print(fail)
file.close()

# sparql.setReturnFormat(N3)
# results = sparql.query().convert()
# g = Graph()
# g.parse(data=results, format="n3")
# print(g.serialize(format='n3'))
# sparql.setReturnFormat(XML)
# results = sparql.query().convert()
# print(results.serialize(format='xml'))
a = 4