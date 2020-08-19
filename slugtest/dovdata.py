import numpy as np
import pandas as pd
from pydov.search.grondwaterfilter import GrondwaterFilterSearch
from pydov.search.boring import BoringSearch
from pydov.search.interpretaties import LithologischeBeschrijvingenSearch
from pydov.search.interpretaties import GecodeerdeLithologieSearch
from pydov.search.interpretaties import InformeleStratigrafieSearch
from pydov.search.interpretaties import FormeleStratigrafieSearch
from pydov.search.interpretaties import InformeleHydrogeologischeStratigrafieSearch
from pydov.search.interpretaties import HydrogeologischeStratigrafieSearch
from pydov.util.location import WithinDistance, Point
from owslib.fes import PropertyIsEqualTo, And
import requests
from bs4 import BeautifulSoup


class DOVPutFilter:

  def __init__(self, putnummer, filternummer):
    self.putnummer = putnummer
    self.filternummer = filternummer
    self.x = None
    self.y = None
    self.maaiveld = None
    self.permkey = None
    self.boornummer = None
    self.hcov = None
    self.peilmetingen = None
    self.element = []
    self.refpunten = None

  def add_element(self, naam, van, tot, diameter):
    element = DOVPutFilterElement(self, naam, van, tot, diameter)
    self.element.append(element)

  def get_element(self, naam):
    return [element for element in self.element if element.naam == naam]

  def get_element_from_depth(self, diepte):
    return [element for element in self.element if element.is_in_interval(diepte)][0]

  def to_dataframe(self):
    columns = ['putnummer', 'filternummer', 'pkey_filter', 'boornummer',
               'x', 'y', 'maaiveld_mTAW', 'HCOV',
               'element', 'van', 'tot', 'lengte', 'diameter']
    df = pd.DataFrame(columns=columns)
    for element in self.element:
      df.loc[len(df)] = [self.putnummer, self.filternummer, self.permkey,
                         self.boornummer, self.x, self.y, self.maaiveld, self.hcov,
                         element.naam, element.van, element.tot, element.lengte, element.diameter]
    return df.sort_values(by='van', ascending=True)


class DOVPutFilterElement:

    def __init__(self, putfilter, naam, van, tot, diameter):
      self.putfilter = putfilter
      self.naam = naam
      self.van = van
      self.tot = tot
      self.diameter = diameter
      self.lengte = self.tot - self.van

    def is_in_interval(self, diepte):
      return self.van < diepte < self.tot


class DOVPutFilterQuery:

    def __init__(self, dov_putfilter, search_radius=25):
        self.putfilter = dov_putfilter
        self.search_radius = search_radius
        self.__xml = None
        self.__query = And([
            PropertyIsEqualTo(
                propertyname='gw_id',
                literal=self.putfilter.putnummer,
            ),
            PropertyIsEqualTo(
                propertyname='filternummer',
                literal=str(self.putfilter.filternummer),
            )
        ])

    def info(self):
        gwfilter = GrondwaterFilterSearch()
        return_fields = ('pkey_filter', 'x', 'y', 'mv_mtaw', 'boornummer', 'aquifer_code')
        df = gwfilter.search(query=self.__query, return_fields=return_fields)
        self.putfilter.x = df['x'][0]
        self.putfilter.y = df['y'][0]
        self.putfilter.maaiveld = df['mv_mtaw'][0]
        self.putfilter.permkey = df['pkey_filter'][0]
        self.putfilter.boornummer = df['boornummer'][0]
        self.putfilter.hcov = df['aquifer_code'][0]

    def peilmetingen(self):
        gwfilter = GrondwaterFilterSearch()
        return_fields = ('pkey_filter', 'datum', 'peil_mtaw', 'betrouwbaarheid', 'filterstatus')
        df = gwfilter.search(query=self.__query, return_fields=return_fields)
        self.putfilter.peilmetingen = df

    def __get_xml(self):
        response = requests.get(self.putfilter.permkey + '.xml')
        if response:
            self.__xml = BeautifulSoup(response.text, 'html.parser')
        else:
            print('Request for putfilter XML failed...')

    def opbouw(self):
        if self.__xml is None:
            self.__get_xml()
        opbouw = self.__xml.find_all("onderdeel")
        for onderdeel in opbouw:
            attributes = ['filterelement', 'van', 'tot', 'binnendiameter']
            attr_types = ['text', 'float', 'float', 'float']
            arguments = []
            for i in range(len(attributes)):
                attr = onderdeel.find(attributes[i])
                if attr is not None:
                    attr = attr.text
                    if attr_types[i] == 'float':
                        attr = float(attr)
                arguments.append(attr)
            self.putfilter.add_element(*arguments)

    def refpunten(self):
        if self.__xml is None:
            self.__get_xml()
        refpunten = self.__xml.find_all("referentiepunt")
        df = pd.DataFrame(columns=['datum', 'refpunt'])
        for refpunt in refpunten:
            datum = np.datetime64(refpunt.find('datum').text)
            meting = float(refpunt.find('meetpunt').text)
            df.loc[len(df)] = [datum, meting]
        self.putfilter.refpunten = df

    def lithologische_beschrijvingen(self):
        interpretatie = LithologischeBeschrijvingenSearch()
        specific_fields = ('beschrijving',)
        return self.__get_interpretatie(interpretatie, specific_fields)

    def gecodeerde_lithologie(self):
        interpretatie = GecodeerdeLithologieSearch()
        specific_fields = ('hoofdnaam1_grondsoort', 'hoofdnaam2_grondsoort',
                           'bijmenging1_grondsoort', 'bijmenging1_hoeveelheid', 'bijmenging1_plaatselijk',
                           'bijmenging2_grondsoort', 'bijmenging2_hoeveelheid', 'bijmenging2_plaatselijk',
                           'bijmenging3_grondsoort', 'bijmenging3_hoeveelheid', 'bijmenging3_plaatselijk')
        return self.__get_interpretatie(interpretatie, specific_fields)

    def informele_stratigrafie(self):
        interpretatie = InformeleStratigrafieSearch()
        specific_fields = ('beschrijving',)
        return self.__get_interpretatie(interpretatie, specific_fields)

    def formele_stratigrafie(self):
        interpretatie = FormeleStratigrafieSearch()
        specific_fields = ('lid1', 'lid2')
        return self.__get_interpretatie(interpretatie, specific_fields)

    def informele_hydrostratigrafie(self):
        interpretatie = InformeleHydrogeologischeStratigrafieSearch()
        specific_fields = ('beschrijving',)
        return self.__get_interpretatie(interpretatie, specific_fields)

    def formele_hydrostratigrafie(self):
        interpretatie = HydrogeologischeStratigrafieSearch()
        specific_fields = ('aquifer',)
        return self.__get_interpretatie(interpretatie, specific_fields)

    def __get_interpretatie(self, interpretatie, specific_fields):
        common_fields = ('pkey_interpretatie', 'pkey_boring', 'Auteurs', 'Datum',
                         'betrouwbaarheid_interpretatie', 'x', 'y', 'Z_mTAW',
                         'diepte_laag_van', 'diepte_laag_tot')
        df = interpretatie.search(
            location=WithinDistance(
                Point(self.putfilter.x, self.putfilter.y),
                distance=self.search_radius
            ),
            return_fields=(common_fields + specific_fields)
        )
        if df.empty:
            return {}
        else:
            return self.__add_boor_and_put_nummer(df)

    def __add_boor_and_put_nummer(self, df):
        boring = BoringSearch()
        dfs = dict(tuple(df.groupby('pkey_interpretatie')))
        for df in dfs.values():
            result = boring.search(
                query=PropertyIsEqualTo(
                    propertyname='pkey_boring',
                    literal=df['pkey_boring'].unique().item()
                ),
                return_fields=('boornummer', 'putnummer')
            )
            df.insert(0, 'boornummer', result['boornummer'][0])
            df.insert(0, 'putnummer', result['putnummer'][0])
            df = df.sort_values(by='diepte_laag_van', ascending=True)
        return dfs
