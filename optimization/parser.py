from pyparsing import *


# parse action -maker
def makeLRlike(numterms):
    if numterms is None:
        # None operator can only by binary op
        initlen = 2
        incr = 1
    else:
        initlen = {0:1,1:2,2:3,3:5}[numterms]
        incr = {0:1,1:1,2:2,3:4}[numterms]

    # define parse action for this number of terms,
    # to convert flat list of tokens into nested list
    def pa(s,l,t):
        t = t[0]
        if len(t) > initlen:
            ret = ParseResults(t[:initlen])
            i = initlen
            while i < len(t):
                ret = ParseResults([ret] + t[i:i+incr])
                i += incr
            return ParseResults([ret])
    return pa

def parse(query):
    # setup a simple grammar for 4-function arithmetic
    varname = Word(alphanums+'_') # CharsNotIn(', ')#
    # integer = Word(nums)
    operand = varname
    # operand = integer | varname


    # opPrec definition with parseAction makeLRlike
    # arith2 = infix_notation(operand, #
    arith2 = operatorPrecedence(operand,
        [
        (None, 2, opAssoc.LEFT, makeLRlike(None)),
        ("~", 1, opAssoc.RIGHT, makeLRlike(0)),
        ("&", 2, opAssoc.LEFT, makeLRlike(2)),
        ("|", 2, opAssoc.LEFT, makeLRlike(2)),
        # (oneOf("* /"), 2, opAssoc.LEFT, makeLRlike(2)),
        # (oneOf("+ -"), 2, opAssoc.LEFT, makeLRlike(2)),
        ])

    return arith2.parseString(query)[0].asList()


def parse2(query):
    varname = Word(alphanums+'_') # CharsNotIn(', ')#
    # integer = Word(nums)
    operand = varname

    expr = infixNotation(
    operand,
        [
            (None, 2, opAssoc.LEFT, makeLRlike(None)),
            ("~", 1, opAssoc.RIGHT, makeLRlike(0)),
            ("&", 2, opAssoc.LEFT, makeLRlike(2)),
            ("|", 2, opAssoc.LEFT, makeLRlike(2)),
        ]
    )
    return expr.parseString(query)[0].asList()


def getSequence(query,i,exp):
    if len(query)==3:
        if isinstance(query[0],list):
            # print(query[0])
            exp,i = getSequence(query[0],i,exp)
            query[0]='s'+ str(i)
            i += 1
        if isinstance(query[2],list):
            # print(query[2])
            exp,i = getSequence(query[2],i,exp)
            query[2] = 's'+ str(i)
            i += 1
        exp += query[0] + query[1] + query[2]+','+'s'+str(i) + '\n'
        # print(exp, i)
        return exp, i
    elif len(query)==2:
        if isinstance(query[1],list):
            exp,i = getSequence(query[1],i,exp)
            query[1]='s'+ str(i)
            i += 1
        exp += query[0]+query[1]
        # print(exp)
        return exp, i

if __name__ == '__main__':


    # query = "car & red | bus & yellow"
    query = "car & red | bus & (yellow | green)"
    # query = '((a|b))'
    print(query)
    result = parse(query)
    print(result)

    # result2 = parse2(query)
    # print('result2',result2)
    # print(getSequence(result2,0,''))

    exp,i = getSequence(result,0,'')
    print(exp)

    # print(arith.parseString("A+B+C*D+E")[0])
    # print(arith.parseString("12AX+34BY+C*5DZ+E")[0])