module PruebasEcuaciones

using Gridap
using GridapGmsh
using Gridap.FESpaces
using Gridap.Geometry
using PyCall
using PyPlot
using JLD
using StatsBase

export simular_camino_par

model = GmshDiscreteModel("geometria.msh")
#writevtk(model,"model")
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V0 = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["Dv1","Dv2"])
degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
neumanntags = ["Neumman"]
Γ = BoundaryTriangulation(model,tags=neumanntags)
dΓ = Measure(Γ,degree)

function solucion_en_malla(Ω,uh)
    ndcel=Ω.grid.cell_node_ids
    ndcoor=Ω.grid.node_coordinates
    u=zeros(length(ndcel))
    sols=uh.cell_dof_values
    for i in 1:length(ndcel)
        for j in 1:length(sols[i])
            u[ndcel[i][j]]=sols[i][j]
        end
    end    
    return ndcoor,u
end

function pasar_a_cartesianas(ndcoor,u,dx,dy)
    x=Int[]
    y=Int[]
    for nod in ndcoor
        append!(x,Int(round(nod[1]/dx))+1)
        append!(y,Int(round(nod[2]/dy))+1)
    end
    matA=zeros(maximum(x),maximum(y))
    for i in 1:length(ndcoor)
        matA[x[i],y[i]]=u[i]
    end
    return matA
end

function resolver_una_etapa_decision(Pec,T_1,T_2,S,vaf,V0,Ω,dΩ,Γ,dΓ,dx,dy,dt,t0,Tf,u0,com)
    g1(x,t::Real)=T_1
    g1(t::Real) = x -> g1(x,t)
    g2(x,t::Real)=T_2
    g2(t::Real) = x -> g2(x,t)
    Ug = TransientTrialFESpace(V0,[g1,g2])

    m(u,v) = ∫( u*v )dΩ
    a(u,v) = (1.0/Pec)*∫( (∇(u)⋅∇(v)) )dΩ +∫( (vaf⋅(∇(u)))*v)dΩ
    b(t,v) = ∫( S(t)*v )dΩ
    op_Af = TransientConstantMatrixFEOperator(m,a,b,Ug,V0)    
    linear_solver = LUSolver()
    Δt = dt
    θ = 0.5
    ode_solver = ThetaMethod(linear_solver,Δt,θ)
    t₀ = t0
    T = Tf
    u₀=interpolate_everywhere(0.0,Ug(0.0))
    if !com
        u₀=u0
    end
    uₕₜ= solve(ode_solver,op_Af,u₀,t₀,T)
    tiempos=Float64[]
    soluciones=Array{Float64}[]
    ndcoor,u=solucion_en_malla(Ω,u₀)
    matA=pasar_a_cartesianas(ndcoor,u,dx,dy)
    append!(soluciones,[matA])
    append!(tiempos,t₀)
    uret=u₀
    tret=0.0
    for (uₕ,t) in uₕₜ
        #ndcoor,u=solucion_en_malla(Ω,uₕ)
        #matA=pasar_a_cartesianas(ndcoor,u,dx,dy)
        #append!(soluciones,[matA])
        #append!(tiempos,t)
        uret=uₕ
        tret=t
    end    
    ndcoor,u=solucion_en_malla(Ω,uret)
    matA=pasar_a_cartesianas(ndcoor,u,dx,dy)
    return tret,[matA],uret
end

function S(x,t,posc)
    c=posc(t)
    if abs(c[1]-x[1])<0.1 && abs(c[2]-x[2])<0.1
        return 200.0
    else
        return 0.0
    end
end

function T1_aux(a)
    if a=="OO" || a=="OF"
        return -0.5
    else
        return 0.0
    end
end

function T2_aux(a)
    if a=="OO" || a=="FO"
        return -0.5
    else
        return 0.0
    end
end

function vax(a,x)
    if a=="OO"
        return VectorValue(0.0,5.0)
    elseif a=="OF"
        if x[1]<0.5
            return VectorValue(0.0,5.0)
        else
            return VectorValue(0.0,0.0)
        end
    elseif a=="FO"
        if x[1]>0.5
            return VectorValue(0.0,5.0)
        else
            return VectorValue(0.0,0.0)
        end
    else
        return VectorValue(0.0,0.0)
    end
end


np = pyimport("numpy")
function simular_camino(Pec,V0,Ω,dΩ,Γ,dΓ,dx,dy,Ndeci,Tf,fdeci)
    t2=LinRange(0.0, Tf, Ndeci+1)
    pol=["FF"]
    x0=rand()
    y0=0.4+0.6*rand()
    posc(t::Real)=(x0-0.1*t,y0-0.1*t)
    St(t::Real)= x -> S(x,t,posc)
    ttodos=Float64[]
    solTodos=Array{Float64,2}[]
    com=true
    u0=Gridap.FESpaces.FEFunction
    for i in 1:Ndeci  
        T1f=T1_aux(pol[i])
        T2f=T2_aux(pol[i])
        vg(x)=vax(pol[i],x) 
        temps,soluciones,usig=resolver_una_etapa_decision(Pec,T1f,T2f,St,vg,V0,Ω,dΩ,Γ,dΓ,dx,dy,0.05,t2[i],t2[i+1],u0,com)
        append!(ttodos,temps)
        append!(solTodos,soluciones)
        u0=usig
        anew=fdeci(soluciones[end])
        append!(pol,[anew])
        com=false
    end
    save("data/solpruba.jld", "sol",solTodos, "t", ttodos, "centroS",posc.(ttodos),"tiemposDeci",t2,"pol",pol)
    return ttodos,solTodos
end

function simular_camino_par(Pec,Ndeci,Tf,fdeci)
    dx,dy=1.0/50.0,1.0/50.0
    return simular_camino(Pec,V0,Ω,dΩ,Γ,dΓ,dx,dy,Ndeci,Tf,fdeci)
end 

end