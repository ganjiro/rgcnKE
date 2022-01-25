import base64
import sparql

result = sparql.query("https://servicemap.disit.org/WebAppGrafo/sparql","""
                      select distinct ?p ?o from  
                  <http://www.disit.org/km4city/resource/Eventi/Eventi_a_Firenze> where 
                  {service <https://servicemap.disit.org/WebAppGrafo/sparql>  
                    {<http://www.disit.org/km4city/schema#Event>?p ?o.
                      }
                    }
                  """)
print('a')