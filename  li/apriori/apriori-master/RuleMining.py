#!/usr/bin/env python3
import csv
from collections import defaultdict
from optparse import OptionParser


class Apriori(object):
    def __init__(self, min_sup, min_conf):
        self.min_sup = min_sup
        self.min_conf = min_conf

    def calc(self, file_path):
        transactions = self.getTransactions(file_path)
        items = self.getItems(transactions)
        items_count = defaultdict(int)
        frequencies = dict()

        self.trans_length = len(transactions)
        self.items = items

        # build 1-term set
        current_frequency = self.getItemsMinSupport(transactions, items,
                                                    items_count, self.min_sup)

        k = 1
        # repeat until current_frequency is empty
        while current_frequency != set():
            frequencies[k] = current_frequency
            k += 1
            upd_items = self.buildItemsSet(
                current_frequency, k)
            current_frequency = self.getItemsMinSupport(transactions, upd_items,
                                                        items_count, self.min_sup)
        self.items_count = items_count
        self.frequencies = frequencies

        return items_count, frequencies

    def buildItemsSet(self, terms, k):
        return set([term1.union(term2) for term1 in terms for term2 in terms
                    if len(term1.union(term2)) == k])

    def getItems(self, transactions):
        items = set()
        for line in transactions:
            for item in line:
                items.add(frozenset([item]))
        return items

    def getTransactions(self, file_path):
        transactions = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for line in reader:
                line_items = []
                for item in line:
                    line_items.append(item)
                transactions.append(line_items)

        headers_set = transactions[0]
        transactions = transactions[1:]
        transactions_pair = []
        for transaction in transactions:
            xx = set([(header, item)
                      for header, item in zip(headers_set, transaction)])
            transactions_pair.append(xx)
        return transactions_pair

    def getItemsMinSupport(self, transactions, items, frequencies, min_sup):
        items_ = set()
        local_set_ = defaultdict(int)
        for item in items:
            frequencies[item] += sum(
                [1 for trans in transactions if item.issubset(trans)])
            local_set_[
                item] += sum([1 for trans in transactions if item.issubset(trans)])

        n = len(transactions)
        for item, cnt in local_set_.items():
            items_.add(item) if float(cnt)/n >= min_sup else None

        return items_

    def getRules(self, rhs):
        rules = dict()
        for key, value in self.frequencies.items():
            for item in value:
                if rhs.issubset(item) and len(item) > 1:
                    item_supp = self.getSupport(item)
                    item = item.difference(rhs)
                    conf = item_supp / self.getSupport(item)
                    if conf >= self.min_conf:
                        rules[item] = (item_supp, conf)
        return rules

    def getSupport(self, item):
        return self.items_count[item] / self.trans_length


if __name__ == '__main__':

    # Parsing command-line parameters
    optParser = OptionParser()
    optParser.add_option('-i', '--input',
                         dest='inputFilePath',
                         help='Input a csv file',
                         type='string',
                         default=None)  # input a csv file

    optParser.add_option('-o', '--output',
                         dest='outputFilePath',
                         help='Output a txt file',
                         type='string',
                         default=None)  # output a txt file

    optParser.add_option('-s', '--min_supp',
                         dest='min_supp',
                         help='Mininum support',
                         type='float',
                         default=0.10)  # mininum support value

    optParser.add_option('-c', '--min_conf',
                         dest='min_conf',
                         help='Mininum confidence',
                         type='float',
                         default=0.40)  # mininum confidence value

    (options, args) = optParser.parse_args()

    input_file_path = options.inputFilePath
    output_file_path = options.outputFilePath
    min_sup = options.min_supp
    min_conf = options.min_conf

    apriori = Apriori(min_sup, min_conf)
    items_count, frequencies = apriori.calc(input_file_path)

    file_out = open(output_file_path, "w")

    file_out.write("Support={}\nConfidence={}\n".
                   format(min_sup, min_conf))

    for key, value in frequencies.items():
        file_out.write('{}-term set:\n'.format(key))
        for items in value:
            file_out.write(str(list(items)) + "\n")
        file_out.write('-'*80)
        file_out.write("\n")

    file_out.write("2-set rules:\n")

    index = 1

    for rhs in apriori.items:
        rules = apriori.getRules(rhs)
        for key, value in rules.items():
            key_list = list(key)[0]
            rhs_list = list(rhs)[0]
            file_out.write("Rule#{}: {{{}={}}} => {{{}={}}}".
                           format(index, key_list[0], key_list[1], rhs_list[0], rhs_list[1]))
            file_out.write("(Support=%.2f, Confidence=%.2f)\n" %
                           (value[0], value[1]))
            index += 1
