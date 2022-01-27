# import requests
#
# '''
# ttl: text/turtle;charset=utf-8
# n3: text/n3; charset=utf-8
# nt: text/plain
# rdf: application/rdf+xml
# owl: application/rdf+xml
# nq: application/n-quads
# trig: application/trig
# jsonld: application/ld+json
# '''
#
# data = open('km4city_test/tutte_raw_turtle.ttl').read()
# print(data)
# headers = {'Content-Type': 'text/turtle;charset=utf-8'}
# r = requests.post('http://localhost:3030/Progetto_KE3/data?default', data=data, headers=headers)
# print(r)


from pyfuseki.rdf import rdf_prefix, NameSpace as ns
from tests.insert_demo import dp, op


@rdf_prefix('http://expample.com/')
class RdfPrefix():
    Person: ns
    Dog: ns

rp = RdfPrefix()

from pyfuseki import FusekiUpdate
from rdflib import Graph, Literal, RDF

g = Graph()

from pyfuseki.rdf import rdf_property
from rdflib import URIRef as uri

@rdf_property('http://example.org/')
class ObjectProperty:
    own: uri

@rdf_property('http://example.org/')
class DataProperty:
    hasName: uri
person = rp.Person['12345']

dog = rp.Dog['56789']

g.add((person, RDF.type, rp.Person.to_uri()))
g.add((dog, RDF.type, rp.Dog.to_uri()))
g.add((person, dp.hasName, Literal('Ryan')))
g.add((dog, dp.hasName, Literal('lucy')))
g.add((person, op.own, dog))

fuseki = FusekiUpdate('http://localhost:3030', 'test_db')
fuseki.insert_graph(g)