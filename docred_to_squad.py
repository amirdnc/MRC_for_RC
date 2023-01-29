import copy
import json
import random
from collections import defaultdict

from tqdm import tqdm

ent_to_rel = {'ORG MISC': ['P607', 'P1056', 'P355', 'P400', 'P136', 'P800', 'P118', 'P37', 'P1344', 'P155', 'P279', 'P156', 'P527', 'P17', 'P361', 'P264', 'P112', 'P463', 'P140', 'P102', 'P1441', 'P127', 'P171', 'P1001', 'P737', 'P27', 'P166', 'P31', 'P749', 'P131', 'P179', 'P364', 'P674', 'P178', 'P495', 'P175', 'P54', 'P276', 'P1366', 'P449', 'P1365', 'P1412', 'P40', 'P50', 'P159', 'P740', 'P150', 'P25', 'P137', 'P1376', 'P194', 'P36', 'P19', 'P20', 'P108', 'P35', 'P170', 'P176', 'P123', 'P488', 'P22', 'P937', 'P272', 'P1336', 'P39', 'P551', 'P172', 'P241'], 'ORG LOC': ['P17', 'P495', 'P159', 'P131', 'P150', 'P1001', 'P30', 'P740', 'P276', 'P37', 'P27', 'P361', 'P527', 'P1366', 'P706', 'P36', 'P355', 'P749', 'P127', 'P112', 'P156', 'P19', 'P20', 'P172', 'P807', 'P364', 'P155', 'P1376', 'P264', 'P710', 'P1344', 'P1365', 'P800', 'P937', 'P206', 'P279', 'P190', 'P1056', 'P840', 'P737', 'P137', 'P31', 'P205', 'P136', 'P140', 'P118', 'P607', 'P463', 'P1412', 'P69', 'P171', 'P108', 'P54', 'P551', 'P176', 'P1336', 'P241', 'P403', 'P102', 'P178', 'P194', 'P123', 'P175', 'P488', 'P22', 'P50', 'P179', 'P1441', 'P449'], 'PER ORG': ['P241', 'P108', 'P264', 'P69', 'P27', 'P102', 'P463', 'P39', 'P54', 'P118', 'P172', 'P140', 'P800', 'P170', 'P178', 'P1441', 'P361', 'P166', 'P607', 'P710', 'P179', 'P449', 'P123', 'P136', 'P272', 'P175', 'P112', 'P20', 'P19', 'P937', 'P194', 'P551', 'P22', 'P127', 'P176', 'P749', 'P400', 'P737', 'P740', 'P156', 'P279', 'P162', 'P31', 'P3373', 'P17', 'P355', 'P527', 'P1412', 'P155', 'P137', 'P26', 'P1344', 'P1365', 'P40', 'P131', 'P1366', 'P674', 'P488', 'P6', 'P35', 'P25', 'P840', 'P86', 'P171', 'P1001', 'P161', 'P495', 'P36', 'P1056', 'P276', 'P159', 'P1376', 'P190', 'P57'], 'PER MISC': ['P607', 'P166', 'P108', 'P1344', 'P1441', 'P1412', 'P674', 'P800', 'P737', 'P27', 'P136', 'P54', 'P118', 'P463', 'P39', 'P156', 'P3373', 'P155', 'P361', 'P279', 'P140', 'P102', 'P527', 'P364', 'P20', 'P40', 'P22', 'P26', 'P840', 'P264', 'P400', 'P488', 'P171', 'P170', 'P25', 'P937', 'P175', 'P19', 'P172', 'P123', 'P179', 'P1365', 'P112', 'P50', 'P31', 'P449', 'P57', 'P69', 'P551', 'P37', 'P1376', 'P272', 'P17', 'P178', 'P495', 'P86', 'P1056', 'P241', 'P161', 'P162', 'P749', 'P127', 'P355', 'P706', 'P676', 'P710', 'P176', 'P1366', 'P35', 'P740', 'P58', 'P276', 'P131', 'P137', 'P1001'], 'PER LOC': ['P27', 'P19', 'P20', 'P17', 'P551', 'P937', 'P159', 'P131', 'P1412', 'P54', 'P172', 'P740', 'P495', 'P1344', 'P102', 'P241', 'P140', 'P108', 'P69', 'P22', 'P26', 'P840', 'P463', 'P800', 'P607', 'P36', 'P127', 'P40', 'P1001', 'P364', 'P276', 'P136', 'P361', 'P30', 'P264', 'P279', 'P1376', 'P25', 'P527', 'P190', 'P3373', 'P1441', 'P706', 'P206', 'P39', 'P137', 'P31', 'P156', 'P150', 'P171', 'P155', 'P710', 'P166', 'P37', 'P118', 'P123', 'P674', 'P176', 'P355', 'P403', 'P737', 'P488', 'P272', 'P112', 'P35', 'P6', 'P178', 'P1366', 'P449', 'P170', 'P1365', 'P179', 'P50', 'P1336', 'P175', 'P205', 'P161', 'P749', 'P807'], 'LOC MISC': ['P1344', 'P37', 'P1441', 'P31', 'P364', 'P463', 'P172', 'P607', 'P361', 'P1376', 'P17', 'P131', 'P137', 'P155', 'P112', 'P1365', 'P527', 'P150', 'P171', 'P279', 'P36', 'P25', 'P1001', 'P206', 'P400', 'P118', 'P1412', 'P127', 'P35', 'P6', 'P40', 'P156', 'P140', 'P136', 'P194', 'P840', 'P276', 'P179', 'P706', 'P123', 'P166', 'P26', 'P205', 'P495', 'P39', 'P1336', 'P1366', 'P800', 'P170', 'P190', 'P749', 'P403', 'P102', 'P1056', 'P50', 'P20', 'P22', 'P355', 'P19', 'P162', 'P449', 'P674', 'P178', 'P159', 'P175', 'P264', 'P737', 'P710', 'P676', 'P272', 'P30', 'P57', 'P3373', 'P176', 'P161', 'P27'], 'LOC LOC': ['P172', 'P131', 'P17', 'P150', 'P30', 'P527', 'P36', 'P1376', 'P37', 'P1336', 'P206', 'P205', 'P276', 'P1001', 'P361', 'P495', 'P1412', 'P27', 'P1366', 'P1365', 'P403', 'P159', 'P155', 'P463', 'P364', 'P706', 'P551', 'P31', 'P137', 'P127', 'P156', 'P807', 'P190', 'P740', 'P279', 'P20', 'P710', 'P840', 'P937', 'P19', 'P112', 'P6', 'P800', 'P194', 'P26', 'P3373', 'P737', 'P176', 'P35', 'P118', 'P140', 'P161', 'P355', 'P749', 'P171', 'P170', 'P1344', 'P108', 'P69', 'P40', 'P25', 'P39', 'P264', 'P22', 'P449', 'P1056', 'P162', 'P1441', 'P50', 'P136', 'P54', 'P179', 'P175', 'P607', 'P166', 'P674', 'P178'], 'MISC MISC': ['P607', 'P179', 'P361', 'P156', 'P171', 'P155', 'P364', 'P527', 'P279', 'P31', 'P449', 'P400', 'P264', 'P495', 'P136', 'P166', 'P176', 'P17', 'P123', 'P1056', 'P178', 'P127', 'P175', 'P710', 'P137', 'P1344', 'P749', 'P272', 'P1441', 'P674', 'P25', 'P40', 'P161', 'P1412', 'P170', 'P131', 'P737', 'P22', 'P463', 'P800', 'P50', 'P1365', 'P1366', 'P37', 'P57', 'P276', 'P840', 'P86', 'P807', 'P112', 'P140', 'P355', 'P58', 'P162', 'P740', 'P676', 'P241', 'P27', 'P159', 'P26', 'P39', 'P118', 'P937', 'P20', 'P706', 'P19', 'P194', 'P1001', 'P3373', 'P172', 'P54', 'P108', 'P1336', 'P36'], 'MISC LOC': ['P17', 'P495', 'P740', 'P364', 'P276', 'P1001', 'P27', 'P131', 'P710', 'P36', 'P840', 'P37', 'P31', 'P159', 'P361', 'P1376', 'P150', 'P30', 'P127', 'P463', 'P1366', 'P172', 'P136', 'P527', 'P937', 'P19', 'P20', 'P279', 'P40', 'P171', 'P194', 'P161', 'P179', 'P264', 'P156', 'P86', 'P205', 'P22', 'P25', 'P123', 'P140', 'P155', 'P176', 'P175', 'P449', 'P178', 'P1412', 'P807', 'P1336', 'P403', 'P674', 'P26', 'P737', 'P206', 'P706', 'P551', 'P272', 'P749', 'P50', 'P800', 'P190', 'P355', 'P137', 'P162', 'P58', 'P57', 'P118', 'P69', 'P1441', 'P170', 'P607', 'P54', 'P112', 'P1365', 'P3373', 'P166', 'P108', 'P676'], 'ORG ORG': ['P137', 'P264', 'P127', 'P749', 'P355', 'P361', 'P17', 'P118', 'P276', 'P1365', 'P194', 'P1001', 'P241', 'P159', 'P176', 'P123', 'P178', 'P140', 'P527', 'P1366', 'P279', 'P102', 'P155', 'P131', 'P31', 'P463', 'P156', 'P449', 'P172', 'P112', 'P136', 'P175', 'P737', 'P150', 'P171', 'P108', 'P272', 'P1056', 'P54', 'P30', 'P1441', 'P69', 'P162', 'P161', 'P807', 'P740', 'P170', 'P495', 'P27', 'P488', 'P710', 'P1336', 'P50', 'P607', 'P551', 'P800', 'P1344', 'P39', 'P3373', 'P26', 'P36', 'P1376', 'P20', 'P403', 'P19', 'P166', 'P400', 'P37', 'P40', 'P22', 'P190', 'P58'], 'ORG TIME': ['P571', 'P576', 'P577', 'P582', 'P580', 'P585', 'P570', 'P569', 'P1198'], 'MISC TIME': ['P577', 'P580', 'P582', 'P585', 'P571', 'P576', 'P449', 'P570', 'P569', 'P1198'], 'MISC ORG': ['P175', 'P264', 'P176', 'P137', 'P449', 'P272', 'P123', 'P178', 'P127', 'P749', 'P710', 'P1001', 'P276', 'P162', 'P102', 'P156', 'P170', 'P140', 'P279', 'P527', 'P31', 'P155', 'P179', 'P241', 'P17', 'P136', 'P361', 'P463', 'P112', 'P131', 'P54', 'P400', 'P171', 'P674', 'P86', 'P108', 'P69', 'P161', 'P495', 'P118', 'P159', 'P676', 'P172', 'P166', 'P355', 'P737', 'P740', 'P1441', 'P1365', 'P1366', 'P607', 'P50', 'P22', 'P58', 'P1412', 'P57', 'P27', 'P40', 'P194', 'P150', 'P36', 'P1376', 'P840', 'P39', 'P37', 'P364', 'P706', 'P488', 'P807', 'P551', 'P1056', 'P1344', 'P25', 'P937'], 'MISC PER': ['P175', 'P162', 'P161', 'P86', 'P57', 'P58', 'P264', 'P676', 'P674', 'P50', 'P170', 'P710', 'P31', 'P527', 'P127', 'P1441', 'P155', 'P123', 'P3373', 'P40', 'P737', 'P156', 'P112', 'P131', 'P179', 'P276', 'P22', 'P25', 'P26', 'P488', 'P361', 'P495', 'P17', 'P27', 'P178', 'P272', 'P171', 'P840', 'P166', 'P449', 'P1366', 'P279', 'P176', 'P19', 'P35', 'P136', 'P137', 'P463', 'P36', 'P364', 'P1001', 'P551', 'P1365', 'P39', 'P108', 'P6', 'P54', 'P1344', 'P800', 'P140', 'P749', 'P102', 'P20', 'P1412', 'P194', 'P1056'], 'PER TIME': ['P569', 'P570', 'P576', 'P571', 'P577', 'P580', 'P582', 'P585'], 'LOC ORG': ['P17', 'P131', 'P194', 'P172', 'P463', 'P137', 'P241', 'P749', 'P30', 'P150', 'P127', 'P361', 'P607', 'P118', 'P1366', 'P1365', 'P527', 'P140', 'P31', 'P36', 'P1376', 'P279', 'P176', 'P35', 'P178', 'P123', 'P155', 'P156', 'P264', 'P1344', 'P807', 'P159', 'P171', 'P1001', 'P190', 'P449', 'P175', 'P69', 'P102', 'P710', 'P54', 'P37', 'P206', 'P166', 'P272', 'P1412', 'P6', 'P112', 'P108', 'P355', 'P27', 'P737', 'P276', 'P1336', 'P170', 'P39', 'P400', 'P495', 'P162', 'P50', 'P179', 'P205', 'P706', 'P740', 'P403', 'P840', 'P40', 'P674', 'P1441'], 'LOC TIME': ['P571', 'P576', 'P585', 'P580', 'P569', 'P577', 'P1198', 'P582', 'P570'], 'LOC PER': ['P127', 'P150', 'P131', 'P35', 'P6', 'P40', 'P26', 'P155', 'P170', 'P27', 'P1376', 'P17', 'P527', 'P22', 'P37', 'P112', 'P276', 'P36', 'P57', 'P361', 'P190', 'P3373', 'P50', 'P674', 'P159', 'P172', 'P710', 'P25', 'P140', 'P175', 'P488', 'P161', 'P1366', 'P706', 'P39', 'P31', 'P58', 'P194', 'P86', 'P676', 'P162', 'P123', 'P156', 'P737', 'P137', 'P179', 'P19', 'P206', 'P279', 'P171', 'P176', 'P178', 'P1365', 'P1412', 'P54', 'P20', 'P1441', 'P800', 'P449', 'P205', 'P749', 'P495', 'P1056', 'P108', 'P840', 'P272', 'P355', 'P463'], 'PER PER': ['P26', 'P22', 'P40', 'P264', 'P3373', 'P25', 'P737', 'P50', 'P170', 'P112', 'P31', 'P161', 'P57', 'P162', 'P800', 'P175', 'P527', 'P58', 'P39', 'P1441', 'P674', 'P123', 'P551', 'P361', 'P463', 'P27', 'P140', 'P19', 'P178', 'P86', 'P54', 'P937', 'P20', 'P279', 'P155', 'P102', 'P127', 'P156', 'P171', 'P607', 'P1344', 'P108', 'P166', 'P676', 'P272', 'P749', 'P355', 'P179', 'P1366', 'P69', 'P172', 'P6', 'P136', 'P1365', 'P35', 'P1412', 'P488', 'P176', 'P131', 'P150', 'P17', 'P710', 'P194', 'P1001', 'P241', 'P190', 'P740', 'P449', 'P1376', 'P36', 'P1056', 'P37', 'P137', 'P495', 'P840', 'P159', 'P706', 'P276'], 'ORG PER': ['P527', 'P488', 'P112', 'P50', 'P170', 'P35', 'P674', 'P175', 'P710', 'P1001', 'P17', 'P161', 'P40', 'P58', 'P31', 'P737', 'P57', 'P6', 'P127', 'P355', 'P155', 'P123', 'P162', 'P150', 'P159', 'P1441', 'P3373', 'P156', 'P178', 'P1366', 'P131', 'P361', 'P676', 'P279', 'P26', 'P749', 'P86', 'P264', 'P463', 'P22', 'P176', 'P25', 'P1365', 'P19', 'P179', 'P740', 'P27', 'P276', 'P136', 'P140', 'P20', 'P1056', 'P1412', 'P1376', 'P171', 'P137', 'P194', 'P551', 'P800', 'P36', 'P272', 'P190', 'P449', 'P108', 'P54', 'P39', 'P118', 'P37', 'P102', 'P495'], 'TIME TIME': ['P155', 'P156', 'P361', 'P527'], 'TIME PER': ['P86'], 'NUM PER': ['P50'], 'TIME LOC': ['P17'], 'LOC NUM': ['P1198', 'P571', 'P582', 'P576'], 'NUM LOC': ['P17'], 'ORG NUM': ['P571', 'P570', 'P1198', 'P576', 'P577', 'P580'], 'MISC NUM': ['P585', 'P571', 'P1198', 'P577', 'P580', 'P582'], 'PER NUM': ['P1198', 'P570', 'P577', 'P569']}


rel_qs = {}
rel_qs['P6'] = ("Who is the head of government of {}?", '{} is the head of government of what country?')   # head of government
rel_qs['P17'] = ("In what country does {} located in?","What is located in country {}?")  # country
rel_qs['P19'] = ("Where was {} born in?","Who was born in {}?")  # place of birth
rel_qs['P20'] = ("Where did {} died?","Who died in {}?")  # place of death
rel_qs['P22'] = ("Who is the father of {}?","Who is the child of {}?")  # Father of  TODO - isn't it too similar?
rel_qs['P25'] = ("Who is the Mother of {}?","Who is the child of {}?")  # mother of 
rel_qs['P26'] = ("Who is the spouse of {}?","Who is the spouse of {}?")  # Spouse of
rel_qs['P27'] = ("{} is the citizen of what country?","Who is the citizen of {}?")  # country of citizenship
rel_qs['P30'] = ("In what continent is {} located in?","In what continent doe {} located in?")  # continent
rel_qs['P31'] = ("{} is an instance of what?","What is an example of {}?")  # instance of
rel_qs['P35'] = ("Who is the head of state of {}?","{} is the head of what state?")  # Head of state
rel_qs['P36'] = ("What is the capitol of {}?","{} is the capitol of what country?")  # capitol
rel_qs['P37'] = ("What is the official language of {}?","{} is the official language of what country?")  # official language TODO: is country correct here?
rel_qs['P39'] = ("What is or was the position held by {}?","Who is holding or held the position of {}?")  # position held
rel_qs['P40'] = ("Who is the child of {}?", "Who is the parent of {}?")  # Child 
rel_qs['P50'] = ("Who is the author of {}?","What did {} wrote?")  # author
rel_qs['P54'] = ("What sport team is {} member of?","Who is the member of the sport team {}?")  # member of sports team
rel_qs['P57'] = ("Who was the director of {}?","What was directed by {}?")  # Director
rel_qs['P58'] = ("Who was the screenwriter of {}?","{} was the screen writer for what play?")  # screenwriter
rel_qs['P69'] = ("Where did {} study?","Who study at {}?")  # educated at 
rel_qs['P86'] = ("What piece of music was written by {}?","Who wrote the musical piece {}?")  # composer
rel_qs['P102'] = ("What political party did {} belong to?","Who is a member of the political party {}?")  # member of political party
rel_qs['P108'] = ("Where does {} work?", "Who work in {}?")  #  employer
rel_qs['P112'] = ("Who founded {}?","What was founded by {}?")  # founded by
rel_qs['P118'] = ("In what league does {} play in?","Who play in league {}?")  # league
rel_qs['P123'] = ("Who published {}?","What was published by {}?")  # publisher
rel_qs['P127'] = ("Who owns {}?","What is owned by {}?")  #  owned by
rel_qs['P131'] = ("What is {} administrative territory?","What is located in the administrative territory {}?")  # located in the administrative territorial entity
rel_qs['P136'] = ("What is the genre of {}?","What have the genre {}?")  # genre
rel_qs['P137'] = ("Who is the operator of {}?","What is operated by {}?")  #  operator
rel_qs['P140'] = ("What is the religion of {}?","Who follows the religion {}?")  # religion
rel_qs['P150'] = ("What is an administrative territory of {}?","In what administrative territory does {} located in?")  # contains administrative territorial entity
rel_qs['P155'] = ("What does follow {}?","What come before {}?")  #  follows
rel_qs['P156'] = ("What come before {}?", "What does follow {}?")  # followed by
rel_qs['P159'] = ("Where is the headquarters of {} located?","headquarters of what organization is located at {}?")  # headquarters location part of
rel_qs["P161"] = ("Who is a cast member at {}", "in what movie did {} play in? ")  # "cast member"
rel_qs["P162"] = ("Who is the producer of {}?", "What was produced by {}?")  # "producer"
rel_qs["P166"] = ("What kind of award did {} received?", "Who received the {} award?")  # "award received"
rel_qs["P170"] = ("Who is the creator of {}?", "What did {} created?")  # "creator"
rel_qs["P171"] = ("What is the taxon of {}?", "what is part of taxon {}?")  # "parent taxon"
rel_qs["P172"] = ("What is the ethnic group of {}", "who is part of the ethnic group {}?")  # "ethnic group"
rel_qs["P175"] = ("Who preformed in {}?", "In what did {} preform?")  # "performer"
rel_qs["P176"] = ("What did {} manufacture?", "What was manufactured by {}?")  # "manufacturer"
rel_qs["P178"] = ("Who developed {}?", "What was developed by {}?")  # "developer"
rel_qs["P179"] = ("What is the series that {} is part of?", "What is part of {}?")  # "series"
rel_qs["P190"] = ("What is the sister city of {}?", "What is the sister city of {}?")  # "sister city"
rel_qs["P194"] = ("What is the legislative body of {}?", "Who legislates over {}?")  # "legislative body"
rel_qs["P205"] = ("What is located at {}?", "In what basin country is {} located?")  #basin country-
rel_qs["P206"] = ("What is located at {}?", "Next to what body of water is {} located?")  #located in or next to body of water
rel_qs["P241"] = ("What is the military branch of {}?", "Who commands {}?")  #military branch
rel_qs["P264"] = ("What is the record label of?", "Who recorded under {}?")  #record label
rel_qs["P272"] = ("What company produced {}?", "What did {} produced?")  #production company
rel_qs["P276"] = ("What is located at {}?", "Where is {} located?")  #location
rel_qs["P279"] = ("what is the subclass of {}?", "what is the superclass of {}?")  #subclass of
rel_qs["P355"] = ("What is the parent organization of {}?", "What is the subsidiary of {}?")  #subsidiary
rel_qs["P361"] = ("What is {} a part of?", "What consists of {}?")  #part of
rel_qs['P364'] = ("In what language was {} originally written?","What was originally written in {}?")  #original language of work
rel_qs['P400'] = ("On what platform was {} developed?","What was developed on {}?")  #platform
rel_qs['P403'] = ("Where does {} flows into?","What flows into {}?")  #mouth of the watercourse
rel_qs['P449'] = ("Where was {} originally broadcasted?","What was originally broadcasted at {}?")  #original network
rel_qs['P463'] = ("Who is a member of the organization {}?","What organization is {} a member of?")  #member of
rel_qs['P488'] = ("Who is the chairperson of {}?","In what organization does {} chairs?")  #chairperson
rel_qs['P495'] = ("Whose country of origin is {}?","What is {} country of origin?")  #country of origin
rel_qs['P527'] = ("What parts does {} have?","What has {} parts?")  #has part
rel_qs['P551'] = ("Who lives in {}?","Where does {} live?")  #residence
rel_qs['P569'] = ("When was {} born?","Who was born on {}?")  #date of birth
rel_qs['P570'] = ("When did {} die?","Who died on {}?")  #date of death
rel_qs['P571'] = ("When did {} come into existence?","What was incepted on {}?")  #inception
rel_qs['P576'] = ("When did {} ceased to exist?","What ceased to exist on {}?")  #dissolved, abolished or demolished
rel_qs['P577'] = ("When was {} published?","What was published on {}?")  #publication date
rel_qs['P580'] = ("When does {} starts?","What starts at {}?")  #start time
rel_qs['P582'] = ("When does {} end?","What ends at {}?")  #end time
rel_qs['P585'] = ("In what point in time did {} take place?","What happened in {}?")  #point in time
rel_qs['P607'] = ("In what conflicts did {} participated?","Who participated in the {} conflict?")  #conflict
rel_qs['P674'] = ("Who are the characters in {}?","Where did {} served as character?")  #characters
rel_qs['P676'] = ("Who wrote the lyrics for {}?","What did {} wrote the lyrics for?")  #lyrics by
rel_qs['P706'] = ("What is located at {}?","Where is {} located?")  #located on terrain feature
rel_qs['P710'] = ("Who is a participant of {}?","What does {} participate in?")  #participant
rel_qs['P737'] = ("Who was influenced by {}?","Who influenced {}?")  #influenced by
rel_qs['P740'] = ("Where was {} formed?","What was formed in {}?")  #location of formation
rel_qs['P749'] = ("What is the parent organization of {}?","What is the subsidiary of {}?")  #parent organization
rel_qs['P800'] = ("What is the notable work of {}?","Whose notable work is {}?")  #notable work
rel_qs['P807'] = ("What was separated from {}?","From what was {} separated?")  #separated from
rel_qs['P840'] = ("Where does {} take place?","What happens at {}?")  #narrative location
rel_qs['P937'] = ("Where does {} work?","Who works at {}?")  #work location
rel_qs['P1001'] = ("To what jurisdiction does {} apply?","Who applies to the jurisdiction of {}?")  #applies to jurisdiction
rel_qs['P1056'] = ("What is the product of material produce of {}?","What country has the product of material produce of {}?")  #product or material produce
rel_qs['P1198'] = ("What is the unemployment rate of {}?","What country has the unemployment rate of {}?")  #unemployment rate
rel_qs['P1336'] = ("What territory did {} claim?", "Who claimed the territory of?")  #territory claimed by
rel_qs['P1344'] = ("Who is a participant of {}?","In what is {} participant of?")  #participant of 
rel_qs['P1365'] = ("Who replaces {}?","Who is replaced by {}?")  #replaces
rel_qs['P1366'] = ("Who is replaced by {}?","Who does {} replace?")  #replaced by
rel_qs['P1376'] = ("What is the capital of {}?","Of what country is {} capital of?")  #capital of
rel_qs['P1412'] = ("Who speaks {}?","What language does {} speak?")  #language spoken, written or signed
rel_qs['P1441'] = ("Who is present in the work {}?","In what work is {} present?") #present in the work
rel_qs['P3373'] = ("Who is the sibling of {}?","Who is the sibling of {}?") #sibling

def make_answers(vs, sentences):
    res = []
    for v in vs:
        cur_s = sentences[:v['sent_id']]
        offset = 0
        for s in cur_s:
            offset += len(join_str(s))
        offset += v['sent_id']  # voodo
        s = sentences[v['sent_id']][:v['pos'][0]]
        offset += len(join_str(s))
        if s and s[-1] != '(' and s[-1] != '-':  # for the space
            offset += 1
        context = ' '.join(join_str(x) for x in sentences)
        v['name'] = join_str(sentences[v['sent_id']][v['pos'][0]:v['pos'][1]])
        # if v['name'] == 'Umbracle':
        if v['name'] != context[offset: offset + len(v['name'])]:
            if v['name'][:4] == context[offset: offset + 4]:
                context = context.replace(context[offset: offset + len(v['name'])], v['name'])
            else:
                offset -= 1
                if v['name'] != context[offset: offset + len(v['name'])]:
                    offset += 2
                if v['name'] != context[offset: offset + len(v['name'])]:
                    offset -= 1

            if v['name'] != context[offset: offset + len(v['name'])]:
                print('real: {}'.format(v['name']))
                print('curr: {}, sen: {}'.format(context[offset: offset + len(v['name'])], v['sent_id']))
        res.append({'text': context[offset: offset + len(v['name'])], 'answer_start': offset})
    return res

def join_str(l):
    s = ' '.join(l)
    s = s.replace('- ', '-')
    s = s.replace(' -', '-')
    s = s.replace(' /', '/')
    s = s.replace('/ ', '/')
    s = s.replace(" '", "'")
    s = s.replace(" ?", "?")
    s = s.replace(" .", ".")
    s = s.replace(" ,", ",")
    s = s.replace(" ;", ";")
    s = s.replace(" :", ":")
    s = s.replace(" )", ")")
    s = s.replace("( ", "(")
    s = s.replace(" n't", "n't")
    s = s.replace("Can not", "Cannot")

    return s

def print_entity_dict(d):
    for d in doc:
        vertexs = d['vertexSet']
        labels = d['labels']
        title = d['title']
        sentences = d['sents']
        p = {}
        p['context'] = ' '.join([join_str(x) for x in sentences])

        for l in labels:
            rel = l['r']
            h_e = vertexs[l['h']][0]['type']
            t_e = vertexs[l['t']][0]['type']
            ner_dict['{} {}'.format(h_e, t_e)].append(rel)

        for v in vertexs:
            names = list(dict.fromkeys(x['name'] for x in v))
            if len(names)>1:
                print('title: {}'.format(title))
                print(names)
                print('len: {}'.format(len(names)))
    for k, v in ner_dict.items():
        ner_dict[k] = list(dict.fromkeys(v))
        v = ner_dict[k]
        print('{} ({}): {}'.format(k, len(v), v))
    print(ner_dict)

def uniq(l):
    return list(dict.fromkeys(l))

def gen_q(heads, tail, is_impossible, id, template, rel):
    res = []
    for head in heads:
        # tail = list(dict.fromkeys(tail))
        q = {}
        q['is_impossible'] = is_impossible
        q['question'] = template.format(head)
        q['subj'] = head
        q['id'] = id
        q['rel'] = rel
        if is_impossible:
            q['answers'] = []
        else:
            q['answers'] = tail
        res.append(q)
    return res

if __name__ == '__main__':


    in_dir = r'/home/nlp/amirdnc/data/docred/'
    out_dir = r'/home/nlp/amirdnc/data/squad2/'
    # in_file = r'train_annotated.json'
    # in_file = r'train_distant.json'
    in_file = r'train.json'
    # out_file = 'doc_trainv3.json'
    out_file = 'tac_marker_tain.json'
    train = True
    # path = r'/home/nlp/amirdnc/data/docred/dev.json'
    in_path = in_dir + in_file
    out_path = out_dir + out_file
    # in_path2 = in_dir + in_file2
    with open(in_path, 'r') as f:
        doc = json.load(f)
    # with open(in_path2, 'r') as f:
    #     doc.extend(json.load(f))

    res = []
    ner_dict = defaultdict(list)
    id = 0
    # print_entity_dict(doc)
    # exit()
    doc = doc[:int(len(doc))]
    doc = doc[:100]  # debug
    t_docs = tqdm(doc)
    for d in t_docs:
        vertexs = d['vertexSet']
        labels = d['labels']
        title = d['title']
        sentences = d['sents']

        ps = []
        for l in labels:
            p = {}
            p['context'] = ' '.join([join_str(x) for x in sentences])
            qas = []
            rel = l['r']
            h_t = vertexs[l['h']][0]['type']
            t_t = vertexs[l['t']][0]['type']
            h_n = vertexs[l['h']][0]['name']
            t_n = vertexs[l['t']][0]['name']
            if '{} {}'.format(h_t, t_t) in ent_to_rel:
                rels = ent_to_rel['{} {}'.format(h_t, t_t)]
            else:
                rels = [rel]
            if train:
                rels = copy.deepcopy(rels)
                random.shuffle(rels)
                if rel in rels:
                    rels.remove(rel)
                    rels = rels[:2]
                    rels.append(rel)
            for r in rels:
                if rel != r:
                    qas.extend(gen_q(uniq(x['name'] for x in vertexs[l['h']]), [], rel != r, id, rel_qs[r][0], r))
                    # qas.extend(gen_q(uniq(x['name'] for x in vertexs[l['h']]), [], rel != r, id, r + '_t {}', r))
                    id += 1
                    qas.extend(gen_q(uniq(x['name'] for x in vertexs[l['t']]), [], rel != r, id, rel_qs[r][1], r))
                    # qas.extend(gen_q(uniq(x['name'] for x in vertexs[l['t']]), [], rel != r, id, r + '_h {}', r))
                    id += 1
                else:
                    qas.extend(
                        gen_q(uniq(x['name'] for x in vertexs[l['h']]), make_answers(vertexs[l['t']], sentences), False, id,
                              rel_qs[r][0], r))
                    id += 1
                    qas.extend(
                        gen_q(uniq(x['name'] for x in vertexs[l['t']]), make_answers(vertexs[l['h']], sentences), False, id,
                              rel_qs[r][1], r))
                    id += 1
                random.shuffle(qas)
                p['qas'] = qas
                ps.append(p)
            qa = {'title': title, 'paragraphs': ps}
        res.append(qa)
    res = {'version': 'v3', 'data': res}
    with open(out_path, 'w') as f:
        json.dump(res, f)

