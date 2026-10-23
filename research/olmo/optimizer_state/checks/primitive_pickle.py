"""Static, deliberately limited pickle data decoder. Never invokes pickle.loads.

Only primitive scalar/container opcodes are accepted. Globals, constructors,
reducers, object building, persistent IDs, extensions and buffers are rejected.
"""
import pickletools, math
def decode(data):
    if len(data)>65536:raise ValueError('Serialized settings exceed 64 KiB')
    stack=[];memo={};mark=object();ops=[];stopped=False
    def marked():
        i=max(i for i,x in enumerate(stack) if x is mark)
        vals=stack[i+1:];del stack[i:];return vals
    for n,(op,arg,pos) in enumerate(pickletools.genops(data)):
        if n>20000:raise ValueError('Too many opcodes')
        name=op.name;ops.append(name)
        if name in ('PROTO','FRAME'):continue
        elif name=='MARK':stack.append(mark)
        elif name in ('NONE','NEWTRUE','NEWFALSE'):stack.append({'NONE':None,'NEWTRUE':True,'NEWFALSE':False}[name])
        elif name in ('BININT','BININT1','BININT2','LONG1','LONG4','INT','LONG','BINFLOAT','FLOAT','SHORT_BINUNICODE','BINUNICODE','BINUNICODE8','UNICODE','SHORT_BINBYTES','BINBYTES','BINBYTES8'):
            if not isinstance(arg,(str,int,float,bytes)):raise ValueError('Nonprimitive scalar')
            if isinstance(arg,float) and not math.isfinite(arg):raise ValueError('Nonfinite scalar')
            stack.append(arg)
        elif name=='EMPTY_LIST':stack.append([])
        elif name=='EMPTY_DICT':stack.append({})
        elif name=='EMPTY_TUPLE':stack.append(())
        elif name=='LIST':stack.append(marked())
        elif name=='TUPLE':stack.append(tuple(marked()))
        elif name in ('TUPLE1','TUPLE2','TUPLE3'):
            n=int(name[-1]);v=stack[-n:];del stack[-n:];stack.append(tuple(v))
        elif name=='APPEND':v=stack.pop();assert type(stack[-1]) is list;stack[-1].append(v)
        elif name=='APPENDS':v=marked();assert type(stack[-1]) is list;stack[-1].extend(v)
        elif name=='DICT':
            v=marked();assert len(v)%2==0;stack.append(dict(zip(v[::2],v[1::2])))
        elif name=='SETITEM':
            v=stack.pop();k=stack.pop();assert type(stack[-1]) is dict;stack[-1][k]=v
        elif name=='SETITEMS':
            v=marked();assert len(v)%2==0 and type(stack[-1]) is dict
            stack[-1].update(zip(v[::2],v[1::2]))
        elif name=='MEMOIZE':memo[len(memo)]=stack[-1]
        elif name in ('BINPUT','LONG_BINPUT','PUT'):memo[int(arg)]=stack[-1]
        elif name in ('BINGET','LONG_BINGET','GET'):stack.append(memo[int(arg)])
        elif name=='STOP':
            if len(stack)!=1 or pos!=len(data)-1:raise ValueError('Bad stop or trailing bytes')
            stopped=True;break
        else:raise ValueError(f'Unsupported opcode {name} at byte {pos}; nothing executed')
        if len(stack)>10000 or len(memo)>10000:raise ValueError('Data structure too large')
    if not stopped:raise ValueError('No final STOP')
    def check(x,seen,depth=0):
        if depth>50:raise ValueError('Too deeply nested')
        if type(x) in (type(None),bool,int,float,str,bytes):return
        if type(x) not in (tuple,list,dict):raise ValueError('Nonprimitive object')
        if id(x) in seen:raise ValueError('Cyclic container')
        seen=seen|{id(x)}
        for v in (list(x.keys())+list(x.values()) if type(x) is dict else x):check(v,seen,depth+1)
    check(stack[0],set())
    return stack[0],sorted(set(ops))

if __name__=='__main__':
    # Handwritten primitive pickle and dangerous GLOBAL counterexample.
    assert decode(b'\x80\x04}\x94(\x8c\x01x\x94G?\xf0\x00\x00\x00\x00\x00\x00u.')[0]=={'x':1.0}
    try:decode(b'cos\nsystem\n.')
    except ValueError:pass
    else:raise AssertionError('Executable opcode was accepted')
    print('Primitive data test passed; executable opcode rejected.')
