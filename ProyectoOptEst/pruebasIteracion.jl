module PruebasIteracion

using PyPlot
using JLD
using LinearAlgebra
using SparseArrays
using BenchmarkTools
import Base.Threads: nthreads, @spawn
using LoopVectorization
using Statistics

export

soluciones=Matrix{Float64}[]
politicas=String[]
Ndeci=1
for nom in readdir("data/soluciones/",join=true)
    sol=load(nom,"sol")
    pol=load(nom,"pol")
    Ndeci=length(pol[2:end])
    append!(soluciones,sol[2:end])
    append!(politicas,pol[2:end])
end

ind=1:length(soluciones)
indsig=[]
for i in ind
    if mod(i,Ndeci)==0
        append!(indsig,i)
    else
        append!(indsig,i+1)
    end
end
indOO=findall(politicas.=="OO")
indOF=findall(politicas.=="OF")
indFO=findall(politicas.=="FO")
indFF=findall(politicas.=="FF")
iOO=1:length(indOO)
iOF=iOO[end]+1:iOO[end]+length(indOF)
iFO=iOF[end]+1:iOF[end]+length(indFO)
iFF=iFO[end]+1:iFO[end]+length(indFF)

indord=cat(indOO,indOF,indFO,indFF,dims=1)
indnew=[findall(indord.==i)[1] for i in ind]
solucionesnew=soluciones[indord]
#indsignew=[findall(indord.==i)[1] for i in indsig]
indsignew=indnew[indsig[indord]]
polnew=politicas[indord];

function rec(T_field,a)
    integral=Float64(sum((abs.(T_field)).>0.5))/Float64(length(T_field))
    c=0.0
    if a=="OF" || a=="FO"
        c+=0.025
    elseif a=="OO"
        c+=2*0.025
    end
    return -1.0*(integral+c)
end

function construir_r(solucionesn,politicasn,indsign)
    r=Float64[]
    @inbounds for i in 1:length(solucionesn)
        append!(r,rec(solucionesn[indsign[i]],politicasn[i]))
    end
    return r
end    

function k_diferencias(X,x)
    s = zero(eltype(x))
    @simd for i in eachindex(x,X)
            s += (X[i] - x[i])^2
        end
    return s
end

function k(X,x,sigma)
    s = zero(eltype(x))
    @simd for i in eachindex(x,X)
            s += (X[i] - x[i])^2
        end
    return exp(-s/sigma)
end

function construir_bloque(soluciones,sigma)
    A=zeros(length(soluciones),length(soluciones))
    @inbounds for i in 1:length(soluciones)
        @inbounds for j in 1:i
            A[j,i]=k_diferencias(soluciones[i],soluciones[j])
        end
    end
    return sparse(Symmetric(A))
end

function construir_bloque_U(sol,sigma)
    A=zeros(length(sol),length(sol))
    @inbounds for i in 1:length(sol)
        @inbounds for j in 1:i
            @inbounds A[j,i]=k_diferencias(sol[i],sol[j])
        end
    end
    return Symmetric(A)
end

function construir_K0(solucionesn,iOO,iOF,iFO,iFF)
    MOO=@views construir_bloque(solucionesn[iOO],sigma)
    MOF=@views construir_bloque(solucionesn[iOF],sigma)
    MFO=@views construir_bloque(solucionesn[iFO],sigma)
    MFF=@views construir_bloque(solucionesn[iFF],sigma)
    return blockdiag(MOO,MOF,MFO,MFF)
end

function construir_KDiag(K,iOO,iOF,iFO,iFF)
    return blockdiag(sparse(K[iOO,iOO]),sparse(K[iOF,iOF]),sparse(K[iFO,iFO]),sparse(K[iFF,iFF]))
end

function construir_K_completa(solucionesn,sigma)
    Kcom=zeros(length(solucionesn),length(solucionesn))
    @inbounds for i in 1:length(solucionesn)
        @inbounds for j in 1:i
             @inbounds Kcom[j,i]= k_diferencias(solucionesn[i],solucionesn[j])
        end
    end
    return Symmetric(Kcom)
end

function construir_K_completa_par(sol1,sol2,sigma)
    Kcom=zeros(length(sol1),length(sol2))
    @inbounds for i in 1:length(sol2)
         @inbounds for j in 1:length(sol1)
            @inbounds Kcom[j,i]=k_diferencias(sol1[j],sol2[i])
        end
    end
    return Kcom
end

function construir_K_completa_parallel(solucionesn,sigma,iOO,iOF,iFO,iFF)
    D1=@spawn construir_bloque_U(solucionesn[iOO],sigma)
    D2=@spawn construir_bloque_U(solucionesn[iOF],sigma)
    D3=@spawn construir_bloque_U(solucionesn[iFO],sigma)
    D4=@spawn construir_bloque_U(solucionesn[iFF],sigma)
    D5=@spawn construir_K_completa_par(solucionesn[iOO],solucionesn[iOF],sigma)
    D6=@spawn construir_K_completa_par(solucionesn[iOF],solucionesn[iFO],sigma)
    D7=@spawn construir_K_completa_par(solucionesn[iFO],solucionesn[iFF],sigma)
    D8=@spawn construir_K_completa_par(solucionesn[iOO],solucionesn[iFO],sigma)
    D9=@spawn construir_K_completa_par(solucionesn[iOF],solucionesn[iFF],sigma)
    D10=@spawn construir_K_completa_par(solucionesn[iOO],solucionesn[iFF],sigma)
    rD1=fetch(D1)
    rD2=fetch(D2)
    rD3=fetch(D3)
    rD4=fetch(D4)
    rD5=fetch(D5)
    rD6=fetch(D6)
    rD7=fetch(D7)
    rD8=fetch(D8)
    rD9=fetch(D9)
    rD10=fetch(D10)
    
    t(A)=transpose(A)
    return Symmetric([rD1 rD5 rD8 rD10;
                    t(rD5) rD2 rD6 rD9;
                    t(rD8) t(rD6) rD3 rD7;
                    t(rD10) t(rD9) t(rD7) rD4
                    ]),blockdiag(sparse(rD1),sparse(rD2),sparse(rD3),sparse(rD4))
end

function U_block_in_place(partA,partSol,sigma)
     @inbounds for i in 1:length(partSol)
         @inbounds for j in 1:i
             @inbounds partA[j,i]=k_diferencias(partSol[i],partSol[j])
            end
        end
end

function full_block_in_place(partA,partSol1,partSol2,sigma)
    @inbounds for i in 1:length(partSol2)
         @inbounds for j in 1:length(partSol1)
             @inbounds partA[j,i]=k_diferencias(partSol1[j],partSol2[i],sigma)
        end
    end
end

function K_com_inplace(solucionesn,sigma,iOO,iOF,iFO,iFF)
    A=zeros(length(solucionesn),length(solucionesn))
    D1=@spawn U_block_in_place(@view(A[iOO,iOO]),deepcopy(solucionesn[iOO]),sigma)
    D2=@spawn U_block_in_place(@view(A[iOF,iOF]),deepcopy(solucionesn[iOF]),sigma)
    D3=@spawn U_block_in_place(@view(A[iFO,iFO]),deepcopy(solucionesn[iFO]),sigma)
    D4=@spawn U_block_in_place(@view(A[iFF,iFF]),deepcopy(solucionesn[iFF]),sigma)
    D5=@spawn full_block_in_place(@view(A[iOO,iOF]),deepcopy(solucionesn[iOO]),deepcopy(solucionesn[iOF]),sigma)
    D6=@spawn full_block_in_place(@view(A[iOF,iFO]),deepcopy(solucionesn[iOF]),deepcopy(solucionesn[iFO]),sigma)
    D7=@spawn full_block_in_place(@view(A[iFO,iFF]),deepcopy(solucionesn[iFO]),deepcopy(solucionesn[iFF]),sigma)
    D8=@spawn full_block_in_place(@view(A[iOO,iFO]),deepcopy(solucionesn[iOO]),deepcopy(solucionesn[iFO]),sigma)
    D9=@spawn full_block_in_place(@view(A[iOF,iFF]),deepcopy(solucionesn[iOF]),deepcopy(solucionesn[iFF]),sigma)
    D10=@spawn full_block_in_place(@view(A[iOO,iFF]),deepcopy(solucionesn[iOO]),deepcopy(solucionesn[iFF]),sigma)
    
    rD1=fetch(D1)
    rD2=fetch(D2)
    rD3=fetch(D3)
    rD4=fetch(D4)
    rD5=fetch(D5)
    rD6=fetch(D6)
    rD7=fetch(D7)
    rD8=fetch(D8)
    rD9=fetch(D9)
    rD10=fetch(D10)
    return Symmetric(A),Symmetric(construir_KDiag(A,iOO,iOF,iFO,iFF))
end

function calcular_producto(X,α,soluciones,sigma)
    v=0.0
    @inbounds for i in 1:length(soluciones)
        v+=α[i]*k(X,soluciones[i],sigma)
    end
    return v
end

function dar_greedy_action(X,α,solucionesn,iOO,iOF,iFO,iFF,sigma)
    vOO=@views calcular_producto(X, α[iOO], solucionesn[iOO],sigma)
    vOF=@views calcular_producto(X, α[iOF], solucionesn[iOF],sigma)
    vFO=@views calcular_producto(X, α[iFO], solucionesn[iFO],sigma)
    vFF=@views calcular_producto(X, α[iFF], solucionesn[iFF],sigma)
    vs=[vOO,vOF,vFO,vFF]
    pos=["OO","OF","FO","FF"]
    ima=argmax(vs)
    return pos[ima],vs[ima]
end

function dar_greedy_action(i::Int,K,α,iOO,iOF,iFO,iFF)
    vOO=@views sum(K[i,iOO].*α[iOO])
    vOF=@views sum(K[i,iOF].*α[iOF])
    vFO=@views sum(K[i,iFO].*α[iFO])
    vFF=@views sum(K[i,iFF].*α[iFF])
    vs=[vOO,vOF,vFO,vFF]
    pos=["OO","OF","FO","FF"]
    ima=argmax(vs)
    return pos[ima],vs[ima]
end

function contruir_vect(solucionesn,indsignew,α,K,iOO,iOF,iFO,iFF)
    vec=zeros(length(solucionesn))
    @inbounds for i in 1:length(solucionesn)
        a,Qm=@views dar_greedy_action(indsignew[i],K,α,iOO,iOF,iFO,iFF)
        vec[i]=Qm
    end
    return vec
end

function iteracion_RFQI(γ,λ,Niter,Kcom,FactLHS,r,solucionesn,indsign,poln,iOO,iOF,iFO,iFF)
    n=length(solucionesn)
    alpha0= zeros(n)
    alpha1=\(FactLHS,r) 
    for i in 1:Niter
        @time r2=contruir_vect(solucionesn,indsign,alpha1,Kcom,iOO,iOF,iFO,iFF)
        alpha0=deepcopy(alpha1)
        alpha1=\(FactLHS,r+γ*r2)
        println("Iteracion: ",i," diferencia : ",sum((alpha0.-alpha1).^2))
    end
    return alpha1
end


end