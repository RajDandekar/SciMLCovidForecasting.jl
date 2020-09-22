#This file was used to generate the .JLD file used in the generate plots code

using MAT

using Plots
using Measures
using Flux
using DifferentialEquations
using DiffEqFlux
using LaTeXStrings
using Random
using Polynomials


vars = matread("C:/Users/Raj/Desktop/2.168/COVID-19-master_latest/COVID-19-master/csse_covid_19_data/csse_covid_19_daily_reports/Rise_Italy_Track.mat")

Infected = vars["Italy_Infected_All"]
Recovered = vars["Italy_Recovered_All"]
Dead= vars["Italy_Dead_All"]
Time = vars["Italy_Time"]

Infected = Infected - Recovered - Dead

#Mobility
#Add apple mobility index data here
vars = matread("C:/Users/Raj/Desktop/Julia Lab Internship/COVID Modelling 1/Italy_Mobility_20June.mat")
Mobility = vars["Mobility"]
Mobility = hcat(Mobility, Mobility[end])
Mobility = Mobility ./ 100

Random.seed!(150)

#Define neural network architecture for contact and recovery rate
ann1 = Chain(Dense(1,10,relu), Dense(10, 10), Dense(10,1))
p1,re = Flux.destructure(ann1)

ann2 = Chain(Dense(3, 10,relu), Dense(10,1))
p1n,re2 = Flux.destructure(ann2)
p3 = [p1; p1n]
ps = Flux.params(p3)

##Define model. Beta (contact rate) is parametrized by the mobility data while recovery rate is not.
function QSIR(du, u, p, t)
    index = Int(floor(t))
    NN1 = abs(re(p[1:141])([Mobility[index]])[1])
    un = [u[1]; u[2]; u[3]]
    NN2 = abs(re2(p[142:192])(un)[1])
    du[1]=  - 0.0003*NN1*u[1]*(u[2])/u0[1]
    du[2] = 0.0003*NN1*u[1]*(u[2])/u0[1] - NN2*u[2]/u0[1]
    du[3] = NN2*u[2]/u0[1]
end


u0 = Float64[60000000.0, 593, 62]
tspan = (1, 95.0)
datasize = 95;
#p = param([0.18])
#p = ([0.2, 0.013, 2])

prob = ODEProblem(QSIR, u0, tspan, p3)
t = range(tspan[1],tspan[2],length=datasize)

###Check if solution is calculated
sol = Array(concrete_solve(prob, Tsit5(),u0, p3, saveat=t))


function predict_adjoint() # Our 1-layer neural network
  Array(concrete_solve(prob,Tsit5(),u0,p3,saveat=t))
end


function loss_adjoint()
 prediction = predict_adjoint()
 #loss = sum(abs2, log.(abs.(Infected[1:end])) .- log.(abs.(prediction[2, :]))) + 1*(sum(abs2, log.(abs.(Recovered[1:end] + Dead[1:end]) ) .- log.(abs.(prediction[3, :] ))))
 loss = sum(abs2, log.(abs.(Infected[1:end])) .- log.(abs.(prediction[2, :]))) + (sum(abs2, log.(abs.(Recovered[1:end] + Dead[1:end]) ) .- log.(abs.(prediction[3, :] ))))
end


Loss = []
P3  =[]
#anim = Animation()
datan = Iterators.repeated((), 20000)
opt = ADAM(0.01)
cb = function()
  display(loss_adjoint())
  global Loss = append!(Loss, loss_adjoint())
  global P3 = append!(P3, p3)

end

cb()


#Number of iterations required = 40000. Can be more depending on model architecture

Flux.train!(loss_adjoint, ps, datan, opt, cb = cb)

####Loss is stagnating around 50! Need to reduce it a lot. How?
L = findmin(Loss)
idx = L[2]
idx1 = (idx-1)*192 +1
idx2 = idx*192
p3n = P3[idx1: idx2]

prediction = Array(concrete_solve(prob,Tsit5(),u0,p3n,saveat=t))

#Compare preeiction with data
bar(Time[1:end], Infected[1:end], alpha = 0.4, xaxis = "Days post 500 infected", yaxis = "Italy: Number of cases", label = "Data: Infected", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :red)
plot!(t, prediction[2, :], xaxis = "Days post 500 infected", yaxis = "Italy: Number of cases", label = "Prediction", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 3, ylims = (0, 20000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
bar!(Time[1:end], Recovered[1:end] + Dead[1:end], alpha = 0.4, xaxis = "Days post 500 infected", yaxis = "Italy: Number of cases", label = "Data: Recovered", legend = :topleft, framestyle = :box, left_margin = 5mm, color = :blue)
plot!(t, prediction[3, :], ylims = (-0.05*maximum(Recovered + Dead),2.5*maximum(Recovered + Dead)), right_margin = 5mm, xaxis = "Days post 500 infected", yaxis = "Italy: Number of cases", label = "Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 3, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
