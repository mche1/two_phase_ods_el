function j=jacobianestim(f,param)
h=1e-10;
fval=f(param);
s=length(fval);
m=length(param);
j=zeros(s,m);
    for(i=1:m)
       param1=param;
       param2=param;
       param1(i)=param(i)+h;
       param2(i)=param(i)-h;
       t=(f(param1)-f(param2))/(2*h);
       j(:,i)=t;
    end
end