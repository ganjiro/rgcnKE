from SPARQLWrapper import SPARQLWrapper, JSON, TSV


def download_km4c(file_path="../subgraph_data.tsv", sparql_endpoint="http://localhost:3030/myFuseki/query", want_tsv=True):
    print("\nAvviata procedura di download del sottografo di km4c...")

    # stabilisce la connessione con l'endpoint
    sparql = SPARQLWrapper(sparql_endpoint)

    # restituisce tutti i km4c:Event (?s)
    sparql.setQuery("""
      PREFIX km4c: <http://www.disit.org/km4city/schema#>
      select distinct ?s from <http://www.disit.org/km4city/resource/Eventi/Eventi_a_Firenze> where {
      service <https://servicemap.disit.org/WebAppGrafo/sparql>  {?s a km4c:Event }
      } 
    """)
    sparql.setReturnFormat(JSON)
    risQuery1 = sparql.query().convert()
    file = open(file_path, "w") # se non esiste lo crea
    if want_tsv:
        file.write('s\tp\to\n')

    print("\nLista entità km4c:Event ottenuta, avvio ricerca triple a massimo 3 hop di distanza...")

    for risQuery1_itr in risQuery1["results"]["bindings"]:  # scorro tutti i km4c:Event
        # restituisce tutte le triple a massimo 3 hop di distanza dalla i-esimo evento
        sparql.setQuery("""
            PREFIX km4c: <http://www.disit.org/km4city/schema#>
            select distinct ?s ?p ?o from  
            <http://www.disit.org/km4city/resource/Eventi/Eventi_a_Firenze> where {
            service <https://servicemap.disit.org/WebAppGrafo/sparql> {
            {?s  ?p ?o. 
                  FILTER(?s = <""" + str(risQuery1_itr["s"]["value"]) + """>)} 
                      union  
                      {?s ?p ?o. <""" + str(risQuery1_itr["s"]["value"]) + """>  ?y ?s.}     
                        union {?s ?p ?o. ?x ?y ?s. <""" + str(risQuery1_itr["s"]["value"]) + """>  ?l ?x.}     
                  }
            }         
        """)
        sparql.setReturnFormat(TSV)
        risQuery2 = sparql.query().convert()


        if want_tsv:  # ritorna le triple in formato TSV
            file.write(risQuery2.decode("UTF-8")[9:-1])  # TSV
            file.write('\n')
        else:  # ritorna le triple in formato NT
            file.write(risQuery2.decode("UTF-8")[9:-1].replace('>\t<', '> <').replace('\n', ' .\n'))  # NT
            file.write(' .\n')

    print("\nTriple a massimo 3 hop di distanza ottenute con successo!")
    print("\nSalvataggio in " + str(file_path))
    if want_tsv:
        print(" in formato .tsv")
    else:
        print(" in formato .nt")
    file.close()


if __name__ == '__main__':
    download_km4c(file_path="../dataset/mio_subgraph_data.TSV", want_tsv=True)
