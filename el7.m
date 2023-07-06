function [param v1 v2]=el7(pop,beta,alpha,theta)
%%%%%%%%%%%%%%%%%%%this is the semi-empirical likelihood estimator 
%%%%%%%%%%%%%%%%%%%ignoring the uncertainty. It is feasible for a 
%%%%%%%%%%%%%%%%%%%large phase 1 sample.
 MaxIter=60;
 MaxStepIter=15;
 MaxTol=1e-5;
 ret_outer=0; %error type
 no_length=0; % # of failure finding step length
 no_improve=0; % # of iter with no improvement
 S=size(pop);
 N=S(1);
 beta_init=beta;
 param=[beta;alpha];
 lb=length(beta);
 lp=length(param);
 paramb=param;
 %ufunc=ufunc_log(pop,param,alpha,theta);
 [lik,u,J,J1,Mb,Mbb]=el7_log_inner(pop,param,lb,theta);
    maxlik=lik;
    ub=u;
    Jb=J;
    for(i=1:MaxIter)
        temp_param=param;
        temp_theta=theta;
        grad=Mb;
        hess=Mbb;
        dir=pinv(hess)*grad;
        step=1;  ind_lik=0;
        for(j=1:MaxStepIter)
            param1=temp_param+dir(1:lp)*step;
            alpha1=param1((lb+1):end);
            theta1=temp_theta+dir((lp+1):end)*step;
%             if(min(alpha1)<0||max(alpha1)>1)
%                 step=step/2;
%            else
                lik1=el7_log_profilelik(pop,param1,lb,theta1);
                if(lik1>maxlik+MaxTol)
                    ind_lik=1;
                    maxlik=lik1;
                    paramb=param1;
                    thetab=theta1;

                end          
                step=step/2;
%            end
        end
        if(ind_lik==0)
          param=param1;
          theta=theta1;
        else
            param=paramb;
            theta=thetab;
        end
                    [lik,u,J,J1,Mb,Mbb]=el7_log_inner(pop,param,lb,theta);
                    ub=u;
                    Jb=J;
        diff=max(abs(param-temp_param));
        if(diff<MaxTol)
             break;
        end
        if (ind_lik==0)
            no_length=no_length+1;  % count failure to find length
                if (no_length==10) % if happens alot, then stop iter
                   ret_outer=2;
                    break;
                end
        end
         %if too many iterations with no improvement of lik, stop###
         if (lik1-lik<MaxTol) 
               no_improve=no_improve+1;
                if (no_improve==20)
                   ret_outer=3;
                    break;
                end
         end

         lik=lik1;
         if (i==MaxIter)
               ret_outer=1;
         end
     end
     if (ret_outer==1)
          fprintf('Maximum Outer Loop Iterations Reached before Convergence \n')
     end
     if (ret_outer==2)
          fprintf('Number of Iterations with No Step Length Exceeds 50 \n')
     end
     if (ret_outer==3)
          fprintf('Number of Iterations with No Increase of Likelihood Exceeds 20 \n')
    end
     v1=diag(inv(J'*inv(u'*u)*J));
     v1=sqrt(v1(1:lb));
     v2=diag(inv(J1'*inv(u'*u)*J1));
     v2=sqrt(v2(1:lb));
    function [lik,u,J,J1,Mb,Mbb]=el7_log_inner(pop,param,lb,theta)
        beta=param(1:lb);
        alpha=param((lb+1):end);
        la=length(alpha);
        uf=ufunc_log(pop,beta,alpha,theta);
        s=eelog(pop,beta,alpha);
        sb=s(:,1:lb);
        sa=s(:,(lb+1):(lb+la));
        sa1=s(:,(lb+la+1):end);
        h=hfunc_log(pop(:,1),pop(:,3),theta);
        
        u=[uf,sb,sa-sa1,h];
          siz=size(u);
          sizu=size(uf);
          sizs=size(s);
          sizh=size(h);
          N1=1/siz(1);
        if(abs(sum(sum(u)))>1e4||isnan(sum(sum(u)))==true)
            lik=-1e5;
            denom=ones(siz(1),1);
            lambda=zeros(siz(2),1);
        else
            siz=size(u);
            N1=1/siz(1);
            lambda0=zeros(siz(2),1);
            opt= optimset('Display','off');
            %den=1-u*lambda0;
            if(isnan(sum(u./repmat(1-u*lambda0,[1,siz(2)])))==true)
                  lik=-1e5;
                  denom=ones(siz(1),1);
                  lambda=zeros(siz(2),1);
            end
           lambda=fsolve(@(lambda)sum(u./repmat(1-u*lambda,[1,siz(2)])),lambda0,opt);
           denom=1-u*lambda;
           if(min(denom)<N1)
                lambda=fmincon(@(lambda)-sum(log(1-u*lambda)),lambda0,u,(1-1/siz(1))*ones(siz(1),1),[],[],[],[],[],opt);
                denom=1-u*lambda;
           end
            lik=-sum(log((1-u*lambda)));
            %paramlik=condl(pop,param,alpha);
            %lik=paramlik+emplik
        end
            J1=jacobianestim(@(x)sum(ufunc_log(pop,x,alpha,theta)./repmat(denom,[1,sizu(2)])),beta);
            J2=jacobianestim(@(x)sum(eelog(pop,x,alpha)./repmat(denom,[1,sizs(2)])),beta);
            J3=zeros(sizh(2),length(beta));%jacobianestim(@(x)sum(eelog(pop,x,alpha)./repmat(denom,[1,sizs(2)])),beta);
   
            J1=[J1;J2(1:(lb+la),:)-[zeros(lb,lb);J2((lb+la+1):(lb+2*la),:)];J3];
            
            J4=jacobianestim(@(x)sum(ufunc_log(pop,beta,x,theta)./repmat(denom,[1,sizu(2)])),alpha);
            J5=jacobianestim(@(x)sum(eelog(pop,beta,x)./repmat(denom,[1,sizs(2)])),alpha);
            J6=zeros(sizh(2),length(alpha));
            
            J2=[J4;J5(1:(lb+la),:)-[zeros(lb,la);J5((lb+la+1):(lb+2*la),:)];J6];
            
            %J=[J1,J2];
            J7=jacobianestim(@(x)sum(ufunc_log(pop,beta,alpha,x)./repmat(denom,[1,sizu(2)])),theta);
            J8=zeros(sizs(2),length(theta));
            J9=jacobianestim(@(x)sum(hfunc_log(pop(:,1),pop(:,3),x)./repmat(denom,[1,sizu(2)])),theta);
            lt=length(theta);
            J3=[J7;J8(1:(lb+la),:)-[zeros(lb,lt);J8((lb+la+1):(lb+2*la),:)];J9];
            J=[J1,J2,J3];
            
%             J11=jacobianestim(@(x)sum(ufunc_log(pop,x,alpha,theta)),beta);
%             J21=jacobianestim(@(x)sum(eelog(pop,x,alpha)),beta);
%             J31=zeros(sizh(2),length(beta));%jacobianestim(@(x)sum(eelog(pop,x,alpha)./repmat(denom,[1,sizs(2)])),beta);
%    
%             J11=[J11;J21([1:lb,(lb+la+1):end],:)-[zeros(lb,lb);J21((lb+1):(lb+la),:)];J31];
%             
%             J41=jacobianestim(@(x)sum(ufunc_log(pop,beta,x,theta)),alpha);
%             J51=jacobianestim(@(x)sum(eelog(pop,beta,x)),alpha);
%             J61=zeros(sizh(2),length(alpha));
%             
%             J21=[J41;J51([1:lb,(lb+la+1):end],:)-[zeros(lb,la);J51((lb+1):(lb+la),:)];J61];
%             
%             %J=[J1,J2];
%             J71=jacobianestim(@(x)sum(ufunc_log(pop,beta,alpha,x)),theta);
%             J81=zeros(sizs(2),length(theta));
%             J91=jacobianestim(@(x)sum(hfunc_log(pop(:,1),pop(:,3),x)),theta);
%             
%             J31=[J71;J81([1:lb,(lb+la+1):end],:)-J81([1:lb,(lb+1):(lb+la)],:);J91];
%             J1=[J11,J21,J31];
            %J=J./repmat(denom,[1,length(lambda)]);
            Mb=J'*lambda;
            u=u./repmat(denom,[1,siz(2)]);
            hess=u'*u;
            Mbb=J'*pinv(hess)*J;
    end
function [lik]=el7_log_profilelik(pop,param,lb,theta)
        beta=param(1:lb);
        alpha=param((lb+1):end);
        la=length(alpha);
        uf=ufunc_log(pop,beta,alpha,theta);
        s=eelog(pop,beta,alpha);
        sb=s(:,1:lb);
        sa=s(:,(lb+1):(lb+la));
        sa1=s(:,(lb+la+1):end);
        h=hfunc_log(pop(:,1),pop(:,3),theta);
        
        u=[uf,sb,sa-sa1,h];
          siz=size(u);
          sizu=size(uf);
          sizs=size(s);
          sizh=size(h);
          N1=1/siz(1);
        if(abs(sum(sum(u)))>1e4||isnan(sum(sum(u)))==true)
            lik=-1e5;
            denom=ones(siz(1),1);
            lambda=zeros(siz(2),1);
        else
            siz=size(u);
            N1=1/siz(1);
            lambda0=zeros(siz(2),1);
             den=1-u*lambda0;
           
            opt= optimset('Display','off');
            lambda=fsolve(@(lambda)sum(u./repmat(1-u*lambda,[1,siz(2)])),lambda0,opt);
            denom=1-u*lambda;
            if(min(denom)<N1)
               lambda=fmincon(@(lambda)-sum(log(1-u*lambda)),lambda0,u,(1-1/siz(1))*ones(siz(1),1),[],[],[],[],[],opt);
                denom=1-u*lambda;
            end
            lik=-sum(log((1-u*lambda)));
            %paramlik=condl(pop,param,alpha);
            %lik=paramlik+emplik
        end

    end
end