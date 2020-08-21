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


class PutFilter:

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
        element = PutFilterElement(self, naam, van, tot, diameter)
        self.element.append(element)

    def get_element(self, naam):
        return [element for element in self.element if element.naam == naam]

    def get_element_from_depth(self, diepte):
        return [element for element in self.element
                if element.is_in_interval(diepte)][0]

    def to_dataframe(self):
        columns = ['putnummer', 'filternummer', 'pkey_filter', 'boornummer',
                   'x', 'y', 'maaiveld_mTAW', 'HCOV',
                   'element', 'van', 'tot', 'lengte', 'diameter']
        df = pd.DataFrame(columns=columns)
        for element in self.element:
          df.loc[len(df)] = [self.putnummer, self.filternummer, self.permkey,
                             self.boornummer, self.x, self.y, self.maaiveld, self.hcov,
                             element.naam, element.van, element.tot, element.lengte,
                             element.diameter]
        return df.sort_values(by='van', ascending=True)


class Interval:

    def __init__(self, van, tot):
        self.van = van
        self.tot = tot

    def is_in_interval(self, diepte):
        return self.van < diepte < self.tot


class PutFilterElement(Interval):

    def __init__(self, putfilter, naam, van, tot, diameter):
        super().__init__(van, tot)
        self.putfilter = putfilter
        self.naam = naam
        self.diameter = diameter
        self.lengte = self.tot - self.van


class Boring:

    def __init__(self, boornummer, putnummer=None, x=None, y=None, maaiveld=None,
                 permkey=None):
        self.boornummer = boornummer
        self.putnummer = putnummer
        self.x = x
        self.y = y
        self.maaiveld = maaiveld
        self.permkey = permkey
        self.interpretatie = []

    def get_interpretatie(self, interpretatie_type):
        return [interpretatie for interpretatie in self.interpretatie
                if interpretatie.type == interpretatie_type]


class Interpretatie:

    def __init__(self, interpretatie_type, boring, permkey=None, datum=None,
                 auteurs=None, betrouwbaarheid=None):
        self.type = interpretatie_type
        self.boring = boring
        boring.interpretatie.append(self)
        self.permkey = permkey
        self.datum = datum
        self.auteurs = auteurs
        self.betrouwbaarheid = betrouwbaarheid
        self.laag = []
        self.__laag_attr = None
        self.__set_laag_attr()

    def __set_laag_attr(self):
        if self.type == 'gecodeerde_lithologie':
            self.__laag_attr = ['hoofdnaam1_grondsoort', 'hoofdnaam2_grondsoort',
                                'bijmenging1_grondsoort', 'bijmenging1_hoeveelheid',
                                'bijmenging1_plaatselijk',
                                'bijmenging2_grondsoort', 'bijmenging2_hoeveelheid',
                                'bijmenging2_plaatselijk',
                                'bijmenging3_grondsoort', 'bijmenging3_hoeveelheid',
                                'bijmenging3_plaatselijk']
        elif self.type == 'formele_stratigrafie':
            self.__laag_attr = ['lid1', 'lid2']
        elif self.type == 'formele_hydrostratigrafie':
            self.__laag_attr = ['aquifer']
        else:  # lithologische_beschrijving of informele_(hydro)stratigrafie
            self.__laag_attr = ['beschrijving']

    def add_laag(self, van, tot, **beschrijving):
        laag = InterpretatieLaag(self, van, tot, **beschrijving)
        self.laag.append(laag)

    def to_dataframe(self):
        columns = ['boornummer', 'putnummer', 'pkey_boring', 'pkey_interpretatie',
                   'datum', 'auteurs', 'x', 'y', 'maaiveld_mTAW', 'betrouwbaarheid',
                   'van', 'tot', 'dikte'] + self.__laag_attr
        df = pd.DataFrame(columns=columns)
        for laag in self.laag:
            rij = [self.boring.boornummer, self.boring.putnummer,
                   self.boring.permkey, self.permkey, self.datum, self.auteurs,
                   self.boring.x, self.boring.y, self.boring.maaiveld,
                   self.betrouwbaarheid, laag.van, laag.tot, laag.dikte]
            for attr in self.__laag_attr:
                rij.append(laag.beschrijving[attr])
            df.loc[len(df)] = rij
        return df.sort_values(by='van', ascending=True)


class InterpretatieLaag(Interval):

    def __init__(self, interpretatie, van, tot, **beschrijving):
        super().__init__(van, tot)
        self.interpretatie = interpretatie
        self.dikte = self.tot - self.van
        self.beschrijving = beschrijving


class PutFilterQuery:

    def __init__(self, putfilter, search_radius=25):
        self.putfilter = putfilter
        self.search_radius = search_radius
        self.boring = []
        self.interpretatie = []
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
        return_fields = ('pkey_filter', 'x', 'y', 'mv_mtaw', 'boornummer',
                         'aquifer_code')
        df = gwfilter.search(query=self.__query, return_fields=return_fields)
        self.putfilter.x = df['x'][0]
        self.putfilter.y = df['y'][0]
        self.putfilter.maaiveld = df['mv_mtaw'][0]
        self.putfilter.permkey = df['pkey_filter'][0]
        self.putfilter.boornummer = df['boornummer'][0]
        self.putfilter.hcov = df['aquifer_code'][0]

    def peilmetingen(self):
        gwfilter = GrondwaterFilterSearch()
        return_fields = ('pkey_filter', 'datum', 'peil_mtaw', 'betrouwbaarheid',
                         'filterstatus')
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
        interpretatie_type = 'lithologische_beschrijving'
        specific_fields = ('beschrijving',)
        return self.__get_interpretatie(interpretatie, interpretatie_type,
                                        specific_fields)

    def gecodeerde_lithologie(self):
        interpretatie = GecodeerdeLithologieSearch()
        interpretatie_type = 'gecodeerde_lithologie'
        specific_fields = ('hoofdnaam1_grondsoort', 'hoofdnaam2_grondsoort',
                           'bijmenging1_grondsoort', 'bijmenging1_hoeveelheid',
                           'bijmenging1_plaatselijk',
                           'bijmenging2_grondsoort', 'bijmenging2_hoeveelheid',
                           'bijmenging2_plaatselijk',
                           'bijmenging3_grondsoort', 'bijmenging3_hoeveelheid',
                           'bijmenging3_plaatselijk')
        return self.__get_interpretatie(interpretatie, interpretatie_type,
                                        specific_fields)

    def informele_stratigrafie(self):
        interpretatie = InformeleStratigrafieSearch()
        interpretatie_type = 'informele_stratigrafie'
        specific_fields = ('beschrijving',)
        return self.__get_interpretatie(interpretatie, interpretatie_type,
                                        specific_fields)

    def formele_stratigrafie(self):
        interpretatie = FormeleStratigrafieSearch()
        interpretatie_type = 'formele_stratigrafie'
        specific_fields = ('lid1', 'lid2')
        return self.__get_interpretatie(interpretatie, interpretatie_type,
                                        specific_fields)

    def informele_hydrostratigrafie(self):
        interpretatie = InformeleHydrogeologischeStratigrafieSearch()
        interpretatie_type = 'informele_hydrostratigrafie'
        specific_fields = ('beschrijving',)
        return self.__get_interpretatie(interpretatie, interpretatie_type,
                                        specific_fields)

    def formele_hydrostratigrafie(self):
        interpretatie = HydrogeologischeStratigrafieSearch()
        interpretatie_type = 'formele_hydrostratigrafie'
        specific_fields = ('aquifer',)
        return self.__get_interpretatie(interpretatie, interpretatie_type,
                                        specific_fields)

    def __get_interpretatie(self, interpretatie, interpretatie_type, specific_fields):
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
        self.interpretatie += self.__dataframe_to_objects(df, interpretatie_type,
                                                          specific_fields)
        return df

    def __dataframe_to_objects(self, result_df, interpretatie_type, specific_fields):
        # split dataframe into multiple dataframes grouped by interpretation
        dfs = dict(tuple(result_df.groupby('pkey_interpretatie')))
        # loop through different interpretation dataframes
        interpretaties = []
        for df in dfs.values():
            # get boring
            boring = self.__get_boring(df)
            # create Interpretatie object
            interpretatie = Interpretatie(
                interpretatie_type=interpretatie_type,
                boring=boring,
                permkey=df['pkey_interpretatie'].unique().item(),
                datum=df['Datum'].unique().item(),
                auteurs=df['Auteurs'].unique().item(),
                betrouwbaarheid=df['betrouwbaarheid_interpretatie'].unique().item()
            )
            df = df.sort_values(by='diepte_laag_van', ascending=True)
            # add InterpretatieLaag objects to Interpretatie object
            for n in range(len(df)):
                beschrijving = {key: df[key].iloc[n] for key in specific_fields}
                interpretatie.add_laag(
                    van=df['diepte_laag_van'].iloc[n],
                    tot=df['diepte_laag_tot'].iloc[n],
                    **beschrijving
                )
            interpretaties.append(interpretatie)
        return interpretaties

    def __get_boring(self, df):
        # query boornummer and putnummer from boring
        pkey_boring = df['pkey_boring'].unique().item()
        boring_search = BoringSearch()
        result = boring_search.search(
            query=PropertyIsEqualTo(
                propertyname='pkey_boring',
                literal=pkey_boring
            ),
            return_fields=('boornummer', 'putnummer')
        )
        boornummer = result['boornummer'].iloc[0]
        # create Boring object if it is not existing yet
        if boornummer not in [boring.boornummer for boring in self.boring]:
            boring = Boring(
                boornummer=result['boornummer'].iloc[0],
                putnummer=result['putnummer'].iloc[0],
                x=df['x'].unique().item(),
                y=df['y'].unique().item(),
                maaiveld=df['Z_mTAW'].unique().item(),
                permkey=pkey_boring
            )
            self.boring.append(boring)
        else:
            boring = [boring for boring in self.boring if boring.boornummer == boornummer][0]
        return boring
