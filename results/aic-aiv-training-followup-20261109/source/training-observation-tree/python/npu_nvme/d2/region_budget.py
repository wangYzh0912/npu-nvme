"""Conservative admission for immutable anchor graphs, pending state and pins."""
import math

def region_budget(*,state_bytes,descriptor_count,page_bytes,retention,active_pins,max_tensor_row_bytes=1024,max_extent_row_bytes=512,control_bytes=0):
    values=(state_bytes,descriptor_count,page_bytes,max_tensor_row_bytes,max_extent_row_bytes)
    if any(type(v) is not int or v<=0 for v in values) or type(active_pins) is not int or active_pins<0 or retention not in (2,3):raise ValueError('budget geometry')
    if page_bytes%4096 or control_bytes<0:raise ValueError('budget alignment')
    if max(max_tensor_row_bytes,max_extent_row_bytes)+128>=page_bytes:raise ValueError('row cannot fit page')
    # Count physical padding per descriptor separately from logical state.
    payload=state_bytes+descriptor_count*4095
    manifest=math.ceil(descriptor_count/( (page_bytes-128)//max_tensor_row_bytes))*page_bytes
    extents=math.ceil(descriptor_count/( (page_bytes-128)//max_extent_row_bytes))*page_bytes
    controls=math.ceil(control_bytes*2/page_bytes)*page_bytes
    generation=payload+manifest+extents+controls
    # Current retention graph + one fallback-only generation + one pending,
    # plus conservative full retained graphs for distinct reader pins.
    physical_generations=retention+2+active_pins*retention
    required=3*4096+physical_generations*generation+2*page_bytes
    return dict(logical_state_bytes=state_bytes,per_generation_upper_bytes=generation,physical_generation_upper_count=physical_generations,required_region_bytes=required,metadata_per_generation_bytes=manifest+extents+controls)
