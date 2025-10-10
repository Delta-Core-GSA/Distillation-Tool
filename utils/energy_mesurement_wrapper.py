from codecarbon import EmissionsTracker
from functools import wraps
from ptflops import get_model_complexity_info
from utils.custom_logger import CustomLogger

#Da usare come:
# @with_emissions_tracking
# def distill(...): ...


# @with_energy_emissions_tracking
# def run_distill():
#     model = distill_model()  # solo distillazione
#     return model

# @with_flops_tracking((3, 224, 224))
# def run_flops():
#     return run_distill()  # restituisce il modello per flops

# Per evitare che il tracciamento energetico includa anche i FLOPs, 
# metti sempre @with_flops_tracking sopra @with_energy_emissions_tracking.


logger = CustomLogger("measurements","txt")

def with_energy_emissions_tracking(func):
    def wrapper(*args, **kwargs):

        role = args[3]

        logger.changeSubject(role)

        tracker = EmissionsTracker(project_name="DistillationBenchmark", measure_power_secs=1) #Aggiungere tag processo
        tracker.start()
        result = func(*args, **kwargs)
        emissions = tracker.stop()

        # Dati finali
        data = tracker.final_emissions_data
        energy_kwh = data.energy_consumed
        energy_cpu_kwh = data.cpu_energy
        energy_gpu_kwh = data.gpu_energy
        energy_ram_kwh = data.ram_energy
        
        # Conversione kWh -> Joule
        energy_joules = energy_kwh * 3.6e6
        energy_cpu_joules = energy_cpu_kwh * 3.6e6
        energy_gpu_joules = energy_gpu_kwh * 3.6e6
        energy_ram_joules = energy_ram_kwh * 3.6e6
        
        # Calcolo della potenza media in kW (energia in kWh / tempo di esecuzione in ore)
        execution_time_seconds = tracker._last_measured_time - tracker._start_time
        execution_time_hours = execution_time_seconds / 3600

        power_kW = energy_kwh / execution_time_hours
        power_cpu_kW = energy_cpu_kwh / execution_time_hours
        power_gpu_kW = energy_gpu_kwh / execution_time_hours
        power_ram_kW = energy_ram_kwh / execution_time_hours

        logger.log_value(value=f"\n[CodeCarbon Report]")
        logger.log_value(value=f"Total energy:      {energy_cpu_kwh:.6f} kWh")
        logger.log_value(value=f"  ├── CPU:         {energy_cpu_kwh:.6f} kWh")
        logger.log_value(value=f"  ├── GPU:         {energy_gpu_kwh:.6f} kWh")
        logger.log_value(value=f"  └── RAM:         {energy_ram_kwh:.6f} kWh\n")

        logger.log_value(value=f"Total energy:      {energy_joules:.0f} J")
        logger.log_value(value=f"  ├── CPU:         {energy_cpu_joules:.0f} J")
        logger.log_value(value=f"  ├── GPU:         {energy_gpu_joules:.0f} J")
        logger.log_value(value=f"  └── RAM:         {energy_ram_joules:.0f} J")

        logger.log_value(value=f"\nAverage Power (during execution):")
        logger.log_value(value=f"  ├── Total Power: {power_kW:.6f} kW")
        logger.log_value(value=f"  ├── CPU Power:   {power_cpu_kW:.6f} kW")
        logger.log_value(value=f"  ├── GPU Power:   {power_gpu_kW:.6f} kW")
        logger.log_value(value=f"  └── RAM Power:   {power_ram_kW:.6f} kW\n")

        logger.log_value(value=f"\nPower consumption percentage (during execution):")
        logger.log_value(value=f"  ├── CPU Power:   {power_cpu_kW/power_kW * 100: .0f} %")
        logger.log_value(value=f"  ├── GPU Power:   {power_gpu_kW/power_kW * 100: .0f} %")
        logger.log_value(value=f"  └── RAM Power:   {power_ram_kW/power_kW * 100: .0f} %\n")

        #Logga i dati in un file CSV
        with open("./logs/emissions_wrapper_log.csv", "a") as f:
            f.write(f"{emissions},{energy_kwh},{energy_cpu_kwh},{energy_gpu_kwh},{energy_ram_kwh}\n")


        return result
    return wrapper


global_tracker = EmissionsTracker(project_name="GlobalBenchmarkTracking", measure_power_secs=1)
global_tracker_started = False

cumulative_energy = {
    "student": 0.0,
    "pretrained_student": 0.0,
    "teacher": 0.0
}

cumulative_energy_j = {
    "student": 0.0,
    "pretrained_student": 0.0,
    "teacher": 0.0
}

runtimes = {
    "student": 0.0,
    "pretrained_student": 0.0,
    "teacher": 0.0
}

def with_global_emission_tracking(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global global_tracker_started

        if not global_tracker_started:
            global_tracker.start()
            global_tracker_started = True

        role = args[3] if len(args) > 3 else kwargs.get('role', 'unknown')
        logger.changeSubject(role)

        global_tracker._measure_power_and_energy()
        energy_before = global_tracker._total_energy.kWh
        timestamp_before = global_tracker._last_measured_time

        result = func(*args, **kwargs)

        global_tracker._measure_power_and_energy()
        energy_after = global_tracker._total_energy.kWh
        timestamp_after = global_tracker._last_measured_time

        delta_energy = energy_after - energy_before
        execution_time_seconds = timestamp_after - timestamp_before
        execution_time_hours = execution_time_seconds / 3600

        # Potenza media
        power_kW = delta_energy / execution_time_hours if execution_time_hours > 0 else 0

        # Energia in Joule
        delta_energy_joules = delta_energy * 3.6e6

        # Accumula per ruolo
        if role not in cumulative_energy:
            cumulative_energy[role] = 0.0
        cumulative_energy[role] += delta_energy
        cumulative_energy_j[role] += delta_energy_joules
        runtimes[role] += execution_time_seconds

        with open("./logs/emissions_wrapper_log.csv", "a") as f:
            f.write(f"{delta_energy},{delta_energy},{0.0},{0.0},{0.0}\n")  # Senza breakdown

        return result
    return wrapper
